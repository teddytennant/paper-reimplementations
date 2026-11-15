"""Load host C++/zlib before test modules import jax."""

from q_deq._host import preload

preload()
