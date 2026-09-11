import hashlib,json,sys,unittest,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from predict_url import CpuPredictor,vector

class RealModelTest(unittest.TestCase):
    def test_url_pipeline_matches_training_reference(self):
        meta=json.loads((ROOT/'artifacts/url_histgb21/features.json').read_text())
        fixtures=json.loads((ROOT/'tests/url_fixtures.json').read_text())
        X=np.concatenate([vector(r['url'],meta['features']) for r in fixtures])
        np.testing.assert_allclose(X,[r['features'] for r in fixtures],rtol=0,atol=1e-6)
        ref=np.array([r['sklearn_score'] for r in fixtures])
        for filename in ['cpu.onnx','gpu.onnx']:
            scores=CpuPredictor(ROOT/'artifacts/url_histgb21'/filename).predict(X)
            np.testing.assert_allclose(scores,ref,atol=1e-6,rtol=0)
            np.testing.assert_array_equal(scores>.5,ref>.5)

    def test_invalid_input_is_rejected(self):
        meta=json.loads((ROOT/'artifacts/url_histgb21/features.json').read_text())
        for value in ['', ' ', 'javascript:alert(1)', 'file:///etc/passwd', 'https://', 'bad host']:
            with self.subTest(value=value),self.assertRaises(ValueError):vector(value,meta['features'])

    def test_model_integrity(self):
        folder=ROOT/'artifacts/url_histgb21'
        manifest=json.loads((folder/'manifest.json').read_text())
        for name,digest in manifest['sha256'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),digest)

    def test_export_preserves_float32_split_boundaries(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from export_histgb import export
        rng=np.random.default_rng(5)
        X=rng.normal(size=(128,3));y=(X[:,0]+X[:,1]>.1).astype(int)
        model=HistGradientBoostingClassifier(max_iter=4,max_leaf_nodes=4,
                                            min_samples_leaf=5,random_state=0).fit(X,y)
        rows=[row for row in X.astype(np.float32)]
        for tree in model._predictors:
            for node in tree[0].nodes:
                if not node['is_leaf']:
                    threshold=np.float32(node['num_threshold'])
                    for value in [threshold,np.nextafter(threshold,np.float32(-np.inf)),
                                  np.nextafter(threshold,np.float32(np.inf))]:
                        row=np.zeros(3,dtype=np.float32);row[node['feature_idx']]=value;rows.append(row)
        values=np.array(rows,dtype=np.float32);reference=model.predict_proba(values)[:,1]
        with tempfile.TemporaryDirectory() as tmp:
            export({'model':model,'features':['a','b','c']},Path(tmp))
            for name in ['cpu.onnx','gpu.onnx']:
                scores=CpuPredictor(Path(tmp)/name).predict(values)
                np.testing.assert_allclose(scores,reference,atol=1e-6,rtol=0)
                np.testing.assert_array_equal(scores>.5,reference>.5)
