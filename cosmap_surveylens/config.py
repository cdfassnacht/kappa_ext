from cosmap.config.models.sky import SkyCoord, Quantity
from cosmap.config.analysis import CosmapAnalysisParameters
from pydantic import Field, BaseModel
from typing import Any

""" 
This file contains the configuration required to run this
analysis. It only contains parameters that are specific to this
analysis. Standard parameters, like "num_threads" or "n_samples" do
not need to be explicitly defined.

We use Pydantic to define the configuration. Pydantic provides type
checking and data validation. This ensures the analysis will
immediately crash if the configuration is invalid. See below for an
example.  
"""


class LensParameters(BaseModel):
    """
    This is an example of parameter which is being validated. The
    source redshift is required to be greater than 0.01, and as a
    result is implicitly required to be a number. The 1.0 is the
    default value.

    Note, we are actually NOT using the astropy SkyCoord type here. We
    use the SkyCoord (and Quantity) types defined by cosmap. This is
    because Pydantic does not support validating non-base types by
    default. The cosmap implementation gives us parsing and validation
    from the config file, and parses them into their corresponding
    astropy types.  When you actually use them in the analysis, they
    will behave exactly as you expect.

    """

    lens_coordinate: SkyCoord
    source_redshift: float = Field(1.0, ge=0.01)


class Filters(BaseModel):
    limiting_magnitude: float | list[float]
    filters: dict = {}

    class Config:
        arbitrary_types_allowed = True


class GeometryParameters(BaseModel):
    radius: Quantity
    inner_radius: Quantity


class Main(CosmapAnalysisParameters):
    """
    The main configurations class for the analysis. Other blocks of
    configuration (like LensParameters) need to be attached to this
    class, and the class must inherit from CosmapAnalysisParameters.

    Within the transformations.json file, the configuration block can
    be accessed by its name within this class. For example, the
    inner_radius parameter can be accessed
    "geometry_parameters.inner_radius"

    """

    n_samples: int = Field(1000, description="Number of samples to draw from the sky")
    lens_parameters: dict[str, LensParameters]
    geometry_parameters: GeometryParameters
    to_remove: dict[str, Any]
    filters: Filters
