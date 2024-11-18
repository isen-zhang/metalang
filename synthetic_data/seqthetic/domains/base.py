from seqthetic.utils import ID

from pydantic import BaseModel, Field
import multiprocessing

from abc import ABC, abstractmethod


class BaseDomainSpec(BaseModel, ABC):
    id: str = ID

    @abstractmethod
    def make_sequences(self, num_token: int, seed: int):
        pass

    def make_sequences_parallel(
        self, num_token: int, seed: int, n_worker: int | None = None
    ):
        n_worker = n_worker or multiprocessing.cpu_count() // 2
        with multiprocessing.Pool(n_worker) as pool:
            # Use pool.map to call self.make_sequences in parallel
            results = pool.starmap(
                self.make_sequences,
                [(num_token // n_worker, seed + i) for i in range(n_worker)],
            )
        # Combine results if needed (e.g., concatenate sequences)
        return [i for j in results for i in j]
