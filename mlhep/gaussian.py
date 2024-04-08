import numpy as np
from pydantic import BaseModel, ConfigDict

class GaussianParameters(BaseModel):
    mu: float
    sigma: float

class GaussianFit(BaseModel):
    observables: np.ndarray
    error: float
    params: GaussianParameters

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def gaussian(self, x):
        mu, sigma = self.params.mu, self.params.sigma
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def check_condition(self):
        x = np.linspace(min(self.observables), max(self.observables), len(self.observables))
        f_x = self.gaussian(x)
        eq = np.abs((self.observables - f_x) / self.error)

        return sum(eq) < len(self.observables)
