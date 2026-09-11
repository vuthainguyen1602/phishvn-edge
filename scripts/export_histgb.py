"""Export a binary numeric HistGB model as CPU trees and TensorRT-compatible tensors.

Requires scikit-learn 1.4.2 for the original trusted joblib artifact. Input contract:
finite float32 features in the saved feature order; categorical/NaN inputs unsupported.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T


def export(bundle, out):
    clf = bundle['model']
    if len(clf.classes_) != 2 or list(clf.classes_) != [0, 1]:
        raise ValueError('Expected binary classes [0, 1]')
    trees = [p[0].nodes for p in clf._predictors]
    if any(np.any(t['is_categorical']) for t in trees):
        raise ValueError('Categorical splits are unsupported')
    ntrees = len(trees)
    branches = max(sum(~t['is_leaf'].astype(bool)) for t in trees)
    leaves = max(sum(t['is_leaf'].astype(bool)) for t in trees)
    indices = np.zeros((ntrees, branches), np.int64)
    thresholds = np.zeros((ntrees, branches), np.float32)
    paths = np.zeros((ntrees, branches, leaves), np.float32)
    counts = np.zeros((ntrees, 1, leaves), np.float32)
    values = np.zeros((ntrees, 1, leaves), np.float32)
    attrs = {k: [] for k in ['nodes_treeids','nodes_nodeids','nodes_featureids','nodes_modes',
             'nodes_values','nodes_truenodeids','nodes_falsenodeids','nodes_missing_value_tracks_true',
             'target_treeids','target_nodeids','target_ids','target_weights']}
    for tid, nodes in enumerate(trees):
        internal = {i:k for k,i in enumerate(np.flatnonzero(~nodes['is_leaf'].astype(bool)))}
        leaf_ids = {i:k for k,i in enumerate(np.flatnonzero(nodes['is_leaf'].astype(bool)))}
        for nid,node in enumerate(nodes):
            threshold = np.float32(node['num_threshold'])
            # Largest representable float32 <= original double threshold preserves <= decisions.
            if float(threshold) > node['num_threshold']:
                threshold = np.nextafter(threshold, np.float32(-np.inf))
            for key,value in [('nodes_treeids',tid),('nodes_nodeids',nid),('nodes_featureids',int(node['feature_idx'])),
                ('nodes_modes','LEAF' if node['is_leaf'] else 'BRANCH_LEQ'),('nodes_values',float(threshold)),
                ('nodes_truenodeids',int(node['left'])),('nodes_falsenodeids',int(node['right'])),
                ('nodes_missing_value_tracks_true',int(node['missing_go_to_left']))]:attrs[key].append(value)
            if node['is_leaf']:
                for key,value in [('target_treeids',tid),('target_nodeids',nid),('target_ids',0),('target_weights',float(node['value']))]:attrs[key].append(value)
            else:
                k=internal[nid];indices[tid,k]=node['feature_idx'];thresholds[tid,k]=threshold
        def visit(nid, route):
            node=nodes[nid]
            if node['is_leaf']:
                lid=leaf_ids[nid];values[tid,0,lid]=node['value']
                for branch,left in route:
                    paths[tid,branch,lid]=1 if left else -1
                    counts[tid,0,lid]+=int(left)
            else:
                visit(int(node['left']),route+[(internal[nid],True)])
                visit(int(node['right']),route+[(internal[nid],False)])
        visit(0,[])
    baseline=np.asarray(clf._baseline_prediction,dtype=np.float32).reshape(1)
    inputs=[h.make_tensor_value_info('features',T.FLOAT,[None,len(bundle['features'])])]
    outputs=[h.make_tensor_value_info('phishing_score',T.FLOAT,[None,1])]
    def save(name,nodes,init,opsets):
        model=h.make_model(h.make_graph(nodes,name,inputs,outputs,init),opset_imports=opsets,ir_version=9)
        onnx.checker.check_model(model);onnx.save(model,out/name)
    save('cpu.onnx',[
        h.make_node('TreeEnsembleRegressor',['features'],['margin'],domain='ai.onnx.ml',n_targets=1,post_transform='NONE',base_values=baseline.tolist(),**attrs),
        h.make_node('Sigmoid',['margin'],['phishing_score'])],[],[h.make_opsetid('',13),h.make_opsetid('ai.onnx.ml',3)])
    init=[nh.from_array(a,n) for n,a in [('indices',indices),('thresholds',thresholds),('paths',paths),('counts',counts),('values',values),('baseline',baseline),('expand',np.array([2],np.int64)),('sum_axes',np.array([1,2,3],np.int64)),('final_shape',np.array([-1,1],np.int64))]]
    save('gpu.onnx',[
        h.make_node('Gather',['features','indices'],['selected'],axis=1),
        h.make_node('LessOrEqual',['selected','thresholds'],['decisions']),
        h.make_node('Cast',['decisions'],['binary'],to=T.FLOAT),
        h.make_node('Unsqueeze',['binary','expand'],['expanded']),
        h.make_node('MatMul',['expanded','paths'],['path_sums']),
        h.make_node('Equal',['path_sums','counts'],['active']),
        h.make_node('Cast',['active'],['active_float'],to=T.FLOAT),
        h.make_node('Mul',['active_float','values'],['weighted']),
        h.make_node('ReduceSum',['weighted','sum_axes'],['sum'],keepdims=0),
        # Reduce trees and leaves independently for each input row.
        h.make_node('Add',['sum','baseline'],['raw']),
        h.make_node('Reshape',['raw','final_shape'],['margin']),
        h.make_node('Sigmoid',['margin'],['phishing_score'])],init,[h.make_opsetid('',13)])
    (out/'features.json').write_text(json.dumps({'features':bundle['features'],'threshold':0.5,'input_dtype':'float32','gpu_batch_size':1},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);export(joblib.load(a.model),out)
