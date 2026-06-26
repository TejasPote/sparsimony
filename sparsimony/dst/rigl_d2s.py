"""Dense-to-Sparse RigL (RigL_D2S).

A thin subclass of :class:`RigL` that trains fully dense for an initial warmup
phase (steps ``< t_dense``) and only then begins RigL dynamic sparse training.

It reuses all of RigL's logic. The only changes are:
  * ``_initialize_masks`` leaves the masks all-ones (dense) at ``prepare()``
    time instead of magnitude-pruning to the target sparsity.
  * ``_step`` defers the one-shot prune-to-target until the dense phase ends,
    reusing the parent ``RigL._initialize_masks`` for that single
    sparsification, then runs RigL's normal steady-state prune+grow updates
    (with dense-gradient regrowth).

Pair this with a :class:`DenseToSparseCosineScheduler`, which returns ``None``
during the dense phase (and suppresses dense-grad accumulation via
``next_step_update``), then a re-anchored cosine prune-ratio afterwards.
"""

from sparsimony.dst.rigl import RigL


class RigL_D2S(RigL):

    def __init__(self, *args, t_dense: int = 0, **kwargs):
        self.t_dense = t_dense
        self._sparse_init_done = False
        super().__init__(*args, **kwargs)

    def _initialize_masks(self) -> None:
        # Dense warmup: record the per-layer target sparsity but leave the masks
        # all-ones so the model trains fully dense until ``t_dense``. The actual
        # magnitude prune is deferred to the first sparse step in ``_step``.
        self._distribute_sparsity(self.sparsity)

    def _step(self) -> bool:
        _topo_updated = False
        self._step_count += 1
        prune_ratio = self.scheduler(self._step_count)
        if prune_ratio is not None:
            if not self._sparse_init_done:
                # First sparse step: one-shot magnitude prune dense -> target,
                # i.e. exactly RigL's original mask initialization. No dense
                # grads are needed for this pure prune.
                super()._initialize_masks()
                self._broadcast_masks()
                self._sparse_init_done = True
                self._logger.info(
                    f"Sparsifying dense LoRA adapters to {self.sparsity} "
                    f"sparsity at step {self._step_count} (end of dense phase)"
                )
            elif self.global_pruning:
                self._global_step(prune_ratio)
            else:
                self._distribute_sparsity(self.sparsity)
                for config in self.groups:
                    if self._is_replica(**config):
                        continue
                    config["prune_ratio"] = prune_ratio
                    config["dense_grads"] = self._get_dense_grads(**config)
                    self.update_mask(**config)
                self._broadcast_masks()
            _topo_updated = True
        if self.scheduler.next_step_update(self._step_count):
            self._accumulate_grads()
        return _topo_updated
