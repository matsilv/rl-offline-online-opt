# Author: Mattia Silvestri

import numpy as np
from tabulate import tabulate

########################################################################################################################


def generate_demands(num_products, observable=None):
    """
    For each product, generate the corresponing demand.
    :param num_products: int; the number of products.
    :param observable: TO-BE DEFINED.
    :return:
    """

    if observable is None:
        demands = np.random.randint(low=1, high=10, size=num_products)
    # NOTE: add demands generation according to some observables
    else:
        raise NotImplementedError()

    return demands

########################################################################################################################


def generate_availability(num_products, num_sets):
    """

    :param num_products: int; number of products.
    :param num_sets: int; number of sets.
    :return:
    """

    feasible = False

    while not feasible:
        availability = np.random.randint(low=0, high=2, size=(num_products, num_sets))

        # Check that all the products are available in at least one set
        available_products = np.sum(availability, axis=1) > 0

        # Check that all the sets have at least one product
        at_least_a_prod = np.sum(availability, axis=0) > 0.

        feasible = available_products.all() and at_least_a_prod.all()

    return availability


########################################################################################################################


class MinSetCover:
    """
    Minimum Set Cover class.

    Attributes:
        seed: int; set seed to ensure reproducibility.
        costs: numpy.array of shape (J, ); cost associated to each set.
        availability: numpy.array of shape (I, J); matrix of {0, 1} representing the availability of the i_th product in
                                                the j_th set.
        demands: numpy.array of shape (J, ); demand associated to each product.
        observable: TO-BE DEFINED.
    """

    def __init__(self, num_sets, num_products, seed=0):
        self._seed = seed
        self._num_sets = num_sets
        self._num_products = num_products

        # Uniform random generation of the cost in the interval [0, 1]
        self._costs = np.random.uniform(low=0, high=1., size=num_sets)

        # Generate demands for each product
        self._demands = generate_demands(num_products=num_products)

        # Randomly generate availability assuring that demands can be satisfied
        self._availability = generate_availability(num_sets=num_sets,
                                                   num_products=num_products)

    @property
    def num_sets(self):
        return self._num_sets

    @property
    def num_products(self):
        return self._num_products

    @property
    def costs(self):
        return self._costs

    @property
    def availability(self):
        return self._availability

    @property
    def demands(self):
        return self._demands

    def __str__(self):
        print_str = ""
        print_str += f"Num. of sets: {self._num_sets} | Num. of products: {self._num_products}\n"
        header = [f"Cost for set n.{idx}" for idx in range(self._num_sets)]
        print_str += tabulate(np.expand_dims(self._costs, axis=0), headers=header, tablefmt='pretty') + '\n'
        header = [f"Demand for product n.{idx}" for idx in range(self._num_products)]
        print_str += tabulate(np.expand_dims(self._demands, axis=0), headers=header, tablefmt='pretty') + '\n'
        header = [f'Availability for set n.{idx}' for idx in range(0, self._num_sets)]
        availability = list()
        for prod_idx in range(0, self._num_products):
            availability.append([f'Product n. {prod_idx}'] + list(self._availability[prod_idx, :]))
        print_str += tabulate(availability, headers=header, tablefmt='pretty') + '\n'

        return print_str

