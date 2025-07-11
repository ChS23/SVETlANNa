import torch
from dataclasses import dataclass
from .element import Element
from .element_parameters import ElementParameters, element_params
from ..simulation_parameters import SimulationParameters
from ..wavefront import Wavefront, mul
from ..parameters import OptimizableTensor
from typing import Iterable
from ..specs import ImageRepr, PrettyReprRepr, ParameterSpecs
from ..visualization import jinja_env, ElementHTML


@element_params('DiffractiveLayer')
@dataclass
class DiffractiveLayerParameters(ElementParameters):
    """
    Parameters for DiffractiveLayer element.
    
    Parameters
    ----------
    mask : OptimizableTensor
        Phase mask for the diffractive layer.
    mask_norm : float, optional
        Normalization value for the phase mask. The phase addition
        is calculated as `2*pi * mask / mask_norm`. Default is 2*pi.
        
    Examples
    --------
    >>> import torch
    >>> mask = torch.rand(100, 100)
    >>> params = DiffractiveLayerParameters(mask=mask, mask_norm=2*torch.pi)
    >>> layer = DiffractiveLayer.from_params(params, simulation_parameters)
    """
    mask: OptimizableTensor
    mask_norm: float = 2 * torch.pi
    
    def validate(self) -> None:
        """Validate diffractive layer parameters."""
        super().validate()
        
        if self.mask_norm <= 0:
            raise ValueError("mask_norm must be positive")
        
        if hasattr(self.mask, 'dim') and self.mask.dim() != 2:
            raise ValueError("mask must be 2-dimensional")


class DiffractiveLayer(Element):
    """A class that described the field after propagating through the
    passive diffractive layer with a given phase mask
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: OptimizableTensor,
        mask_norm: float = 2 * torch.pi
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters
        mask : OptimizableTensor
            Phase mask
        mask_norm : float, optional
            This value will be used as following:
            the phase addition is equal to `2*torch.pi * mask / mask_norm`.
            By default, `2*torch.pi`
        """

        super().__init__(simulation_parameters)

        self.mask = self.process_parameter('mask', mask)
        self.mask_norm = self.process_parameter('mask_norm', mask_norm)

    @classmethod
    def from_params(
        cls, 
        params: DiffractiveLayerParameters, 
        simulation_parameters: SimulationParameters
    ) -> 'DiffractiveLayer':
        """
        Create DiffractiveLayer from parameters.
        
        Parameters
        ----------
        params : DiffractiveLayerParameters
            Parameters for the diffractive layer.
        simulation_parameters : SimulationParameters
            Simulation parameters for the optical system.
            
        Returns
        -------
        DiffractiveLayer
            Created diffractive layer element.
            
        Examples
        --------
        >>> import torch
        >>> mask = torch.rand(100, 100)
        >>> params = DiffractiveLayerParameters(mask=mask)
        >>> layer = DiffractiveLayer.from_params(params, sim_params)
        """
        params.validate()
        return cls(simulation_parameters, **params.to_kwargs())

    @property
    def transmission_function(self) -> torch.Tensor:
        return torch.exp(
            (2j * torch.pi / self.mask_norm) * self.mask
        )

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        """Method that calculates the field after propagating through the SLM

        Parameters
        ----------
        input_field : Wavefront
            Field incident on the SLM

        Returns
        -------
        Wavefront
            The field after propagating through the SLM
        """
        return mul(
            incident_wavefront,
            self.transmission_function,
            ('H', 'W'),
            self.simulation_parameters
        )

    def reverse(self, transmission_wavefront: Wavefront) -> Wavefront:
        """Method that calculates the field after passing the SLM in back
        propagation

        Parameters
        ----------
        transmitted_field : Wavefront
            Field incident on the SLM in back propagation
            (transmitted field in forward propagation)

        Returns
        -------
        Wavefront
            Field transmitted on the SLM in back propagation
            (incident field in forward propagation)
        """
        return mul(
            transmission_wavefront,
            torch.conj(self.transmission_function),
            ('H', 'W'),
            self.simulation_parameters
        )

    def to_specs(self) -> Iterable[ParameterSpecs]:
        mask = self.mask.numpy(force=True)
        mask_min = mask.min()
        mask_max = mask.max()

        return [
            ParameterSpecs(
                'mask', [
                    PrettyReprRepr(self.mask),
                    ImageRepr((255 * (mask - mask_min) / (mask_max - mask_min)).astype('uint8')),
                ]
            ),
            ParameterSpecs(
                'mask_norm', [
                    PrettyReprRepr(self.mask_norm)
                ]
            )
        ]

    @staticmethod
    def _widget_html_(
        index: int,
        name: str,
        element_type: str | None,
        subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template('widget_diffractive_layer.html.jinja').render(
            index=index, name=name, subelements=subelements
        )
