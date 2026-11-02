# Bundled BrainWeb solver

`model.py` is an unchanged copy of the MIT-licensed
[BrainWeb model source](https://github.com/arvidl/brainweb_fisher_kpp/blob/81ecd42e6f37692a711208caa8d24361904f7e65/src/brainweb_kpp/model.py)
at commit `81ecd42e6f37692a711208caa8d24361904f7e65`.

SHA256: `2c6da092c0b4e4eb6871094b3826485ba45b6887452094aa2bae17c6ba316e50`.
The original copyright notice is retained in [LICENSE](LICENSE).

Only the numerical model is bundled. The educational adapter in
[growth_coupled.py](../../growth_coupled.py) uses this repository's existing
checksum-verified BrainWeb inputs; it does not import, download or require a
companion checkout. Its visualizations and controls are local teaching code.
