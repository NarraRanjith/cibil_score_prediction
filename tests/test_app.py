import os
import sys
import pytest

# Ensure repository root is on sys.path so tests can import top-level modules like `app.py`.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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
import os
import sys
import pytest

# Ensure repository root is on sys.path so tests can import top-level modules like `app.py`.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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


