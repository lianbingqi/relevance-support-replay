import numpy as np, torch
from src.agents.context_encoder import TransitionHistoryEncoder,HistoryDynamicsPredictor,NoHistoryDynamicsPredictor
from src.representation.history import transition_feature,current_feature

def test_features():
    s=np.array([1,2,.1,-.2],dtype=np.float32); ns=np.array([1.2,1.9,.2,-.1],dtype=np.float32)
    x=transition_feature(s,2,ns,5); c=current_feature(s,2,5)
    assert x.shape==(13,) and c.shape==(9,) and np.allclose(x[-4:],ns-s)

def test_shapes():
    h=torch.randn(7,8,13); c=torch.randn(7,9)
    e=TransitionHistoryEncoder(13,32,16); assert e(h).shape==(7,16)
    m=HistoryDynamicsPredictor(13,9,4,32,16,64); p,z=m(h,c); assert p.shape==(7,4) and z.shape==(7,16)
    b=NoHistoryDynamicsPredictor(9,4,64); assert b(c).shape==(7,4)
