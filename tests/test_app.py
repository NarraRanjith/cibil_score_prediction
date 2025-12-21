import pytest

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_homepage(client):
    resp = client.get('/')
    assert resp.status_code == 200
    assert b"Predict" in resp.data or b"Please Fill The Information" in resp.data
