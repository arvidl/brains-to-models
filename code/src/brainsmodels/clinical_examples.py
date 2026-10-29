"""Reproduce three didactic plots. No patient data; deterministic examples."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brainsmodels.paper_examples import (
    SHORTCUT_AGREEMENTS, STABLE_ACCURACY, hopfield_12, shortcut_dependence,
)

def main():
    ROOT=Path(__file__).resolve().parents[3]
    OUT=ROOT/'paper'/'figures'; OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    BLUE='#167184'; ORANGE='#ae5e32'; INK='#183143'
    def save(fig,name):
     fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    # Analytic shortcut example: two fixed binary rules; no trained model.
    shortcut=shortcut_dependence(); a=shortcut.agreement
    fig,ax=plt.subplots(figsize=(9,4.6))
    ax.plot(a,shortcut.marker_accuracy,label='Site-marker rule',color=ORANGE,lw=3)
    ax.axhline(STABLE_ACCURACY,label='Stable image-feature rule',color=BLUE,lw=3)
    ax.scatter([.95,.5,.05],[.95,.5,.05],c=ORANGE,s=60,zorder=5)
    for x,y,txt in [(.95,.95,'Training-like'),(.5,.5,'Independent'),(.05,.05,'Reversed')]:
     ax.annotate(txt,(x,y),xytext=(-90 if x>.8 else 12, -27 if y>.8 else 16),textcoords='offset points',fontsize=12)
    ax.set(xlabel='P(site marker agrees with label) in the test population',ylabel='Expected accuracy',xlim=(0,1),ylim=(0,1.04))
    ax.legend(loc='upper left',frameon=False); ax.grid(alpha=.15)
    fig.tight_layout();save(fig,'shortcut')
    # Bayes example. Fixed sensitivity and specificity; varying prevalence.
    p=np.linspace(.001,.5,500); sens=.9;spec=.9
    ppv=sens*p/(sens*p+(1-spec)*(1-p))
    fig,ax=plt.subplots(figsize=(9,4.6));ax.plot(100*p,100*ppv,color=BLUE,lw=3)
    for q in [.01,.1,.5]:
     v=sens*q/(sens*q+(1-spec)*(1-q));ax.scatter([100*q],[100*v],color=ORANGE,s=60,zorder=5)
     ax.annotate(f'{q:.0%} prevalence: {v:.1%} PPV',(100*q,100*v),xytext=(12,-10) if q<.5 else (-190,-30),textcoords='offset points',fontsize=12)
    ax.set(xlabel='Prevalence (%)',ylabel='Positive predictive value (%)',xlim=(0,53),ylim=(0,100));ax.grid(alpha=.15)
    fig.tight_layout();save(fig,'ppv')
    # Classical asynchronous Hopfield network, one stored pattern, 12 units.
    hopfield=hopfield_12()
    pattern=hopfield.pattern; states=hopfield.states; energy=hopfield.energies.tolist()
    assert np.all(np.diff(energy)<=1e-10)
    assert np.array_equal(states[-1],pattern)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={'width_ratios':[1.2,1]})
    ax.step(range(len(energy)),energy,where='post',color=BLUE,lw=3);ax.set(xlabel='Single-unit updates',ylabel='Hopfield energy (arbitrary units)');ax.grid(alpha=.15)
    bx.imshow(np.stack([states[0],states[-1],pattern]),cmap='coolwarm',vmin=-1,vmax=1,aspect='auto',interpolation='nearest');bx.set(yticks=[0,1,2],yticklabels=['Noisy cue','Retrieved','Stored'],xlabel='Unit index',xticks=[0,3,6,9,11],xticklabels=[1,4,7,10,12]);bx.tick_params(length=0);bx.set_title('Blue: -1    Red: +1',fontsize=12)
    fig.tight_layout();save(fig,'hopfield')
    (OUT/'computed_values.json').write_text(json.dumps({'shortcut_accuracy':{**SHORTCUT_AGREEMENTS,'stable_rule':STABLE_ACCURACY},'ppv':{str(q):sens*q/(sens*q+(1-spec)*(1-q)) for q in [.01,.1,.5]},'hopfield_energy':energy,'hopfield_retrieves_pattern':True},indent=2))
    print('Wrote three PDF/PNG figures and computed_values.json')

if __name__ == '__main__':
    main()
