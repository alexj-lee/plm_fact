import pytest
import plm_fact as plm
import numpy as np


# @pytest.fixture
def test_load_fn_rosetta():
    # should split this up
    train_loader, test_loader = plm.data.get_data_loaders_rosetta()
    assert len(train_loader) > 0, 'Train loader has no length'
    assert len(test_loader) > 0, 'Test loader has no length'

    loader_iter = iter(train_loader)
    first = next(loader_iter).compute()
    assert abs(first.mean()) > 0, 'Batch mean was zero'
    second = next(loader_iter).compute()

    assert (
        np.allclose(first, second) is False
    ), 'First and second rows are same'

    batch = np.vstack((first, second))
    assert batch.shape == (2, 1280), 'Batch dimensions are incorrect'

    # return batch  # maybe use this later in fixture
