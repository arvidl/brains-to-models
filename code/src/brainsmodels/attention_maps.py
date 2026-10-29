"""Transparent Transformer attention sublayer: maps are different quantities.

Run: python -m brainsmodels.attention_maps
Only NumPy and Matplotlib are required. No downloads, API calls, or patient data.
The weights are hand specified, NOT trained. This is one self-attention head
with a residual path and scalar readout, not a complete Transformer model.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]

def parameters():
    # X columns: intensity, row position, column position, CLS indicator.
    Wq = np.zeros((4, 2)); Wq[3] = [1., .5]
    Wk = np.array([[2., 0.], [-1., 0.], [-1., 0.], [0., 0.]])
    Wv = np.eye(4)
    Wo = np.eye(4)
    readout = np.array([3., .4, -.2, 0.])
    return Wq, Wk, Wv, Wo, readout

def tokens(patches):
    r, c = np.mgrid[0:4, 0:4]
    X = np.zeros((17, 4), dtype=np.float64)
    X[0, 3] = 1.  # Fixed summary (CLS) token; no image intensity.
    X[1:, 0] = patches.ravel()
    X[1:, 1] = r.ravel()/3
    X[1:, 2] = c.ravel()/3
    return X

def forward(X, params):
    Wq, Wk, Wv, Wo, w = params
    Q, K, V = X@Wq, X@Wk, X@Wv
    S = Q@K.T/np.sqrt(Q.shape[1])
    E = np.exp(S-S.max(axis=1, keepdims=True))
    A = E/E.sum(axis=1, keepdims=True)
    Y = X + A@V@Wo
    return float(Y[0]@w), (Q, K, V, A)

def gradient(X, params):
    # Exact reverse-mode chain rule for f=(X + AVWo)[0] @ w.
    Wq, Wk, Wv, Wo, w = params
    _, (Q, K, V, A) = forward(X, params)
    dY = np.zeros_like(X); dY[0] = w
    dZ = dY@Wo.T
    dA = dZ@V.T
    dV = A.T@dZ
    dS = A*(dA-(dA*A).sum(axis=1, keepdims=True))
    dQ = dS@K/np.sqrt(Q.shape[1])
    dK = dS.T@Q/np.sqrt(Q.shape[1])
    return dY + dQ@Wq.T + dK@Wk.T + dV@Wv.T

def maps(X, params, baseline=0.):
    score, (_, _, _, A) = forward(X, params)
    g = gradient(X, params)[1:, 0].reshape(4, 4)
    effects = []
    for j in range(1, 17):
        altered = X.copy(); altered[j, 0] = baseline
        effects.append(score-forward(altered, params)[0])
    # CLS attention to itself is omitted without renormalizing the other weights.
    return score, A, g, np.array(effects).reshape(4, 4)

def main():
    out = ROOT/'paper'/'figures'; out.mkdir(exist_ok=True)
    patches = np.full((4, 4), .1)
    patches[1:3, 1:3] = .7  # Central synthetic pattern.
    patches[0, 0] = 1.     # Outside-pattern site marker.
    X = tokens(patches); p = parameters()
    score, A, g, occ = maps(X, p)
    # Independent finite-difference verification for all input dimensions.
    eps = 1e-6; numerical = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        plus, minus = X.copy(), X.copy()
        plus[idx] += eps; minus[idx] -= eps
        numerical[idx] = (forward(plus,p)[0]-forward(minus,p)[0])/(2*eps)
    error = float(np.max(np.abs(numerical-gradient(X,p))))
    assert error < 1e-7, error
    assert np.allclose(A.sum(axis=1), 1.)
    # A value-only change leaves A unchanged but changes f.
    p_value = list(p); p_value[2] = p_value[2].copy(); p_value[2][0,0] *= -1
    value_score, (_, _, _, value_A) = forward(X, p_value)
    assert np.array_equal(A, value_A)
    assert not np.isclose(score, value_score)
    # Exactly preserve the central pattern while replacing the marker.
    no_marker = X.copy(); no_marker[1, 0] = .1
    marker_score = forward(no_marker, p)[0]
    assert np.array_equal(X[[6,7,10,11]], no_marker[[6,7,10,11]])
    # Randomization illustrates parameter dependence; no training-label test.
    rng = np.random.default_rng(2026)
    random_p = tuple(rng.normal(0, .5, a.shape) for a in p)
    random_score, random_A, random_g, random_occ = maps(X, random_p)
    _, _, _, occ_alt = maps(X, p, baseline=.1)
    result = dict(score=score, marker_replaced_score=marker_score,
        marker_effect=score-marker_score, value_changed_score=value_score,
        randomized_score=random_score, gradient_check_max_abs_error=error,
        attention_to_cls=float(A[0,0]), attention=A.tolist(),
        signed_gradient=g.tolist(), zero_occlusion=occ.tolist(),
        background_occlusion=occ_alt.tolist(), randomized_gradient=random_g.tolist(),
        numpy_version=np.__version__, matplotlib_version=matplotlib.__version__)
    (out/'attention_maps_values.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':9, 'axes.titlesize':10})
    fig, axes = plt.subplots(2,3,figsize=(10.5,6),layout='constrained')
    panels = [(patches,'Synthetic input','gray',False),
        (A[0,1:].reshape(4,4),'CLS-to-patch attention','viridis',False),
        (g,'Signed input gradient','coolwarm',True),
        (occ,'Occlusion effect (baseline 0)','coolwarm',True),
        (occ_alt,'Occlusion effect (baseline 0.1)','coolwarm',True),
        (random_g,'Gradient after randomization','coolwarm',True)]
    for ax,(data,title,cmap,signed) in zip(axes.flat,panels):
        lim=max(float(abs(data).max()),1e-10)
        im=ax.imshow(data,cmap=cmap,vmin=-lim if signed else 0,vmax=lim)
        ax.set_title(title); ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xlabel('Patch column'); ax.set_ylabel('Patch row')
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Same input, different questions: a hand-specified attention sublayer')
    fig.savefig(out/'attention_maps.pdf'); fig.savefig(out/'attention_maps.png',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    im=ax.imshow(A,cmap='viridis'); ax.set_xlabel('Key position (0 = CLS)')
    ax.set_ylabel('Query position (0 = CLS)'); ax.set_title('Full 17 x 17 attention matrix')
    fig.colorbar(im,ax=ax,label='Weight'); fig.savefig(out/'attention_matrix.png',dpi=160)
    plt.close(fig)
    print(json.dumps({k:result[k] for k in ['score','marker_replaced_score','marker_effect',
        'value_changed_score','randomized_score','gradient_check_max_abs_error']},indent=2))

if __name__ == '__main__':
    main()
