import numpy as np
from src.representation.history import dynamics_transition_feature,transition_feature,current_feature

def test_dynamics_history_feature_excludes_absolute_state():
    s1=np.asarray([1.,2.,.1,-.2],dtype=np.float32)
    ns1=np.asarray([1.2,1.9,.2,-.1],dtype=np.float32)
    shift=np.asarray([10.,-5.,3.,7.],dtype=np.float32)
    s2=s1+shift; ns2=ns1+shift
    d1=dynamics_transition_feature(s1,2,ns1,5)
    d2=dynamics_transition_feature(s2,2,ns2,5)
    assert d1.shape==(9,)
    assert np.allclose(d1,d2)
    assert not np.allclose(transition_feature(s1,2,ns1,5),transition_feature(s2,2,ns2,5))
    assert current_feature(s1,2,5).shape==(9,)
