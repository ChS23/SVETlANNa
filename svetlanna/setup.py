from typing import Iterable, Union
from .elements.element import Element
from .elements.element_parameters import ElementParameters
from .simulation_parameters import SimulationParameters
from .specs import ParameterSpecs, SubelementSpecs
from torch import nn
from torch import Tensor
from warnings import warn
from .visualization import jinja_env, ElementHTML


class LinearOpticalSetup(nn.Module):
    """
    A linear optical network composed of Element's.
    
    Supports both Element instances and ElementParameters for convenient setup creation.
    When ElementParameters are provided, simulation_parameters must also be specified.
    """
    
    def __init__(
        self, 
        elements: Iterable[Union[Element, ElementParameters]], 
        simulation_parameters: SimulationParameters | None = None
    ) -> None:
        """
        Create a linear optical setup.
        
        Parameters
        ----------
        elements : Iterable[Union[Element, ElementParameters]]
            A sequence of optical elements or element parameters.
            If ElementParameters are provided, simulation_parameters must be specified.
        simulation_parameters : SimulationParameters | None, optional
            Required when elements contains ElementParameters.
            Used to create Element instances from ElementParameters.
            
        Examples
        --------
        >>> # Traditional approach with Element instances
        >>> setup = LinearOpticalSetup([
        ...     FreeSpace(sim_params, distance=0.1, method='AS'),
        ...     DiffractiveLayer(sim_params, mask=torch.rand(100, 100))
        ... ])
        >>> 
        >>> # New approach with ElementParameters
        >>> setup = LinearOpticalSetup([
        ...     FreeSpaceParameters(distance=0.1, method='AS'),
        ...     DiffractiveLayerParameters(mask=torch.rand(100, 100))
        ... ], simulation_parameters=sim_params)
        >>> 
        >>> # Mixed approach
        >>> setup = LinearOpticalSetup([
        ...     FreeSpace(sim_params, distance=0.1, method='AS'),  # Element
        ...     DiffractiveLayerParameters(mask=torch.rand(100, 100))  # Parameters
        ... ], simulation_parameters=sim_params)
        """
        super().__init__()

        elements_list = list(elements)
        
        has_parameters = any(isinstance(elem, ElementParameters) for elem in elements_list)
        
        if has_parameters and simulation_parameters is None:
            raise ValueError(
                "simulation_parameters is required when elements contains ElementParameters"
            )
        
        created_elements = []
        for elem in elements_list:
            if isinstance(elem, ElementParameters):
                if not hasattr(type(elem), 'element_class') or type(elem).element_class is None:
                    raise ValueError(
                        f"{type(elem).__name__} must set element_class class attribute"
                    )
                
                element_class = type(elem).element_class()
                created_element = element_class.from_params(elem, simulation_parameters)
                created_elements.append(created_element)
            elif isinstance(elem, Element):
                created_elements.append(elem)
            else:
                raise TypeError(
                    f"Each element must be either Element or ElementParameters, "
                    f"got {type(elem)}"
                )

        self.elements = created_elements
        self.net = nn.Sequential(*created_elements)

        # Validate simulation parameters consistency
        if len(created_elements) > 0:
            first_sim_params = created_elements[0].simulation_parameters

            def check_sim_params(element: Element) -> bool:
                return element.simulation_parameters is first_sim_params

            if not all(map(check_sim_params, self.elements)):
                warn(
                    "Some elements have different SimulationParameters "
                    "instance. It is more convenient to use "
                    "the same SimulationParameters instance."
                )

        # Setup reverse propagation if supported
        if all((hasattr(el, 'reverse') for el in self.elements)):

            class ReverseNet(nn.Module):
                def forward(self, Ein: Tensor) -> Tensor:
                    for el in reversed(created_elements):
                        Ein = el.reverse(Ein)
                    return Ein

            self._reverse_net = ReverseNet()
        else:
            self._reverse_net = None

    def forward(self, input_wavefront: Tensor) -> Tensor:
        """
        A forward function for a network assembled from elements.

        Parameters
        ----------
        input_wavefront : torch.Tensor
            A wavefront that enters the optical network.

        Returns
        -------
        torch.Tensor
            A wavefront after the last element of
            the network (output of the network).
        """
        return self.net(input_wavefront)

    def stepwise_forward(self, input_wavefront: Tensor):
        """
        Function that consistently applies forward method of each element
        to an input wavefront.

        Parameters
        ----------
        input_wavefront : torch.Tensor
            A wavefront that enters the optical network.

        Returns
        -------
        str
            A string that represents a scheme of a propagation through a setup.
        list(torch.Tensor)
            A list of an input wavefront evolution
            during a propagation through a setup.
        """
        this_wavefront = input_wavefront
        # list of wavefronts while propagation of an initial wavefront through the system
        steps_wavefront = [this_wavefront]  # input wavefront is a zeroth step

        optical_scheme = ''  # string that represents a linear optical setup (schematic)

        self.net.eval()
        for ind_element, element in enumerate(self.net):
            # for visualization in a console
            element_name = type(element).__name__
            optical_scheme += f'-({ind_element})-> [{ind_element + 1}. {element_name}] '
            # TODO: Replace len(...) with something for Iterable?
            if ind_element == len(self.net) - 1:
                optical_scheme += f'-({ind_element + 1})->'
            # element forward
            this_wavefront = element.forward(this_wavefront)
            steps_wavefront.append(this_wavefront)  # add a wavefront to list of steps

        return optical_scheme, steps_wavefront

    def reverse(self, Ein: Tensor) -> Tensor:
        if self._reverse_net is not None:
            return self._reverse_net(Ein)
        raise TypeError(
            'Reverse propagation is impossible. '
            'All elements should have reverse method.'
        )

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        return (
            SubelementSpecs(str(i), element) for i, element in enumerate(self.elements)
        )

    @staticmethod
    def _widget_html_(
        index: int,
        name: str,
        element_type: str | None,
        subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template('widget_linear_setup.html.jinja').render(
            index=index, name=name, subelements=subelements
        )
