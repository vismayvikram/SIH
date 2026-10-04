from fastapi.testclient import TestClient

from backend.api import app


def test_dashboard_landing_page_is_served_for_root():
    client = TestClient(app)

    response = client.get('/')

    assert response.status_code == 200
    text = response.text
    assert 'Geospatial Feature Review Demo' in text
    assert 'Open Lalpur Demo' in text
    assert 'Create Project' in text
    assert 'id="create-project-dialog"' in text
    assert 'id="dashboard-project-list"' in text
    assert 'id="nav-projects"' in text
    create_button = text.split('id="btn-create-project"', 1)[1].split('>', 1)[0]
    assert 'disabled' not in create_button
    assert 'type="file"' in text
