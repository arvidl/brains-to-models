"""Reproduce the original attention and Hopfield calculations in v0.2."""
from pathlib import Path
import json
import numpy as np
from brainsmodels.paper_examples import hopfield_12, hopfield_energy, two_key_attention

def main():
    original=two_key_attention(); changed=two_key_attention(values=(2.,-10.))
    attention={'weights':original.weights.tolist(),'output':original.output,
               'changed_value_output':changed.output}
    hopfield=hopfield_12(); xi=hopfield.pattern; W=hopfield.weights
    energy=lambda x:hopfield_energy(x,W)
    trajectory=hopfield.energies.tolist()
    for step,(i,h) in enumerate(zip(hopfield.units,hopfield.fields)):
        before,s=hopfield.states[step:step+2]
        delta=trajectory[step+1]-trajectory[step]
        assert np.isclose(delta,-(s[i]-before[i])*h)
        assert delta<=1e-12
    assert np.array_equal(hopfield.states[-1],xi)
    assert np.isclose(energy(xi),energy(-xi))
    assert np.array_equal(np.sign(W@(-xi)),-xi)
    out={'attention':attention,'hopfield_energy':trajectory,
         'pattern_energy':energy(xi),'inverse_pattern_energy':energy(-xi)}
    path=Path(__file__).resolve().parents[3]/'paper'/'figures'/'appendix_values.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__ == '__main__':
    main()
