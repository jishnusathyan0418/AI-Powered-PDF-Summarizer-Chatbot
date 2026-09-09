"""Browser smoke test against server.py on localhost:8000 (requires playwright)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_local import make_pdf
from playwright.sync_api import sync_playwright, expect

artifact_dir = Path('.cache')
artifact_dir.mkdir(exist_ok=True)
fixture = artifact_dir / 'sample-handbook.pdf'
fixture.write_bytes(make_pdf('Northstar employees receive 23 days of annual leave. The project codename is ORCHID-742.'))
errors = []
checks = []
with sync_playwright() as p:
    browser = p.chromium.launch(channel='msedge', headless=True)
    page = browser.new_page(viewport={'width': 1100, 'height': 850})
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:8000', wait_until='networkidle')
    expect(page.locator('#upload-button')).to_be_visible()
    expect(page.locator('#send-button')).to_be_disabled()
    checks.append('Home loads; chat disabled before upload')

    page.locator('#file-upload').set_input_files({'name': 'broken.pdf', 'mimeType': 'application/pdf', 'buffer': b'not a pdf'})
    expect(page.locator('#message-list')).to_contain_text('not a valid PDF')
    expect(page.locator('#upload-button')).to_be_enabled()
    checks.append('Invalid upload displays error and permits retry')

    page.locator('#file-upload').set_input_files(str(fixture))
    expect(page.locator('#message-list')).to_contain_text('Your PDF is ready', timeout=180000)
    expect(page.locator('#send-button')).to_be_enabled()
    page.locator('#message-input').fill('How many days of annual leave do Northstar employees receive?')
    with page.expect_response('**/process-message', timeout=90000) as received:
        page.locator('#send-button').click()
    response = received.value
    payload = response.json()
    if response.status == 200:
        assert '23' in payload['botResponse'], payload
        checks.append('LIVE Groq answer correctly reports 23 days')
        page.locator('#message-input').fill('What is the project codename?')
        with page.expect_response('**/process-message', timeout=90000) as received:
            page.locator('#message-input').press('Enter')
        second = received.value
        assert second.status == 200 and 'ORCHID-742' in second.json()['botResponse']
        checks.append('LIVE Groq second answer correctly reports ORCHID-742')
    else:
        assert response.status == 503 and 'GROQ_API_KEY' in payload['botResponse'], payload
        expect(page.locator('#message-list')).to_contain_text('GROQ_API_KEY')
        checks.append('Missing Groq key gives a clear error; live answer test BLOCKED')
    expect(page.locator('#send-button')).to_be_enabled()

    # A simulated response verifies rendering only, not answer generation.
    page.route('**/process-message', lambda route: route.fulfill(json={'botResponse': '<img src=x onerror=alert(1)> plain text'}))
    page.locator('#message-input').fill('Test response rendering')
    page.locator('#send-button').click()
    expect(page.locator('#message-list')).to_contain_text('<img src=x onerror=alert(1)> plain text')
    assert page.locator('#message-list img').count() == 0
    page.unroute('**/process-message')
    checks.append('Simulated model HTML is rendered as text')

    page.locator('#light-dark-mode-switch').check()
    expect(page.locator('body')).to_have_class('dark-mode')
    page.locator('#light-dark-mode-switch').uncheck()
    checks.append('Theme toggle works')
    page.screenshot(path=str(artifact_dir / 'chatbot-browser-test.png'), full_page=True)
    page.locator('#reset-button').click()
    expect(page.locator('#upload-button')).to_be_enabled()
    expect(page.locator('#send-button')).to_be_disabled()
    assert page.request.post('http://127.0.0.1:8000/process-message', data={'userMessage': 'Old document?'}).status == 400
    page.locator('#file-upload').set_input_files(str(fixture))
    expect(page.locator('#send-button')).to_be_enabled(timeout=90000)
    checks.append('Reset clears backend state and allows another upload')
    page.locator('#reset-button').click()
    expect(page.locator('#upload-button')).to_be_enabled()
    assert not errors, errors
    checks.append('No browser JavaScript exceptions')
    browser.close()
print(json.dumps(checks, indent=2))
