- Train LoRA says the `sd-scripts` submodule was never checked out, and points at the worker log,
  when that checkout failed, instead of a bare "Script not found". A missing or broken `.venv-exec`
  is named as such rather than reported as missing dependencies in `.venv`.
