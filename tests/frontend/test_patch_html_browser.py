"""Optional real-browser checks; install Playwright and Chromium to run them.

    python -m pip install playwright
    python -m playwright install chromium
    python -m pytest tests/frontend/test_patch_html_browser.py

Neither Playwright nor its browser is required by the core test suite.
"""

import pytest

from lightstim.ir.qec_patch import QECPatch


playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect
pytestmark = pytest.mark.integration


class BrowserPatch(QECPatch):
    def _process_params(self):
        pass

    def build(self):
        self.add_qubit(0, 0, "data", uid=2)
        self.add_qubit(2, 0, "data", uid=8)
        self.add_qubit(1, 1, "data", uid=13)
        self.stabilizers = [
            {"name": "Triangle check", "pauli": {2: "X", 8: "X", 13: "X"}},
            {"name": "Edge check", "pauli": {2: "Z", 8: "Z"}},
        ]
        self.logical_ops = [{"name": "Logical Y", "pauli": {8: "Y", 13: "Y"}}]
        self.num_logicals = 1


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as engine:
        try:
            browser = engine.chromium.launch()
        except playwright.Error as exc:
            if "Executable doesn't exist" in str(exc):
                pytest.skip("Optional Chromium browser missing; run python -m playwright install chromium")
            raise
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    page = browser.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    yield page
    page.close()
    assert not errors, f"Viewer JavaScript errors: {errors}"


@pytest.fixture
def viewer(page, tmp_path):
    path = BrowserPatch().visualize().save_html(tmp_path / "patch.html")
    page.goto(path.as_uri())
    expect(page.locator("#operator-name")).to_have_text("Triangle check")
    return page


def assert_selected(viewer, name, pauli, category):
    expect(viewer.locator(".op.selected")).to_have_count(1)
    expect(viewer.locator(".op.selected .op-name")).to_have_text(name)
    expect(viewer.locator('.op[aria-pressed="true"]')).to_have_count(1)
    expect(viewer.locator("#operator-name")).to_have_text(name)
    expect(viewer.locator("#figure-title")).to_have_text(f"{name} · Pauli support")
    expect(viewer.locator("#category-label")).to_have_text(category)
    expect(viewer.locator("#pauli")).to_have_text(pauli)
    expect(viewer.locator("#weight")).to_be_visible()


def test_category_change_with_no_matches_then_clear_search(viewer):
    """The reported sequence must never leave a logical under stabilizer rows."""
    viewer.get_by_role("button", name="Logicals 1", exact=True).click()
    assert_selected(viewer, "Logical Y", "+Y8 Y13", "Logical representative")
    viewer.get_by_role("searchbox").fill("no such operator")
    viewer.get_by_role("button", name="Stabilizers 2", exact=True).click()
    expect(viewer.locator("#operator-name")).to_have_text("No matching operators")
    expect(viewer.locator("#detail-meta")).to_contain_text("No stabilizers match this filter")
    viewer.get_by_role("searchbox").fill("")
    assert_selected(viewer, "Triangle check", "+X2 X8 X13", "Stabilizer generator")
    expect(viewer.locator("#operator-list .op")).to_have_count(2)
    expect(viewer.locator("#patch-diagram .q title")).to_have_text([
        "q2: (0, 0), data, X support",
        "q8: (2, 0), data, X support",
        "q13: (1, 1), data, X support",
    ])


def test_search_selects_visible_operator_and_preserves_it_when_cleared(viewer):
    viewer.get_by_role("searchbox").fill("z8")
    expect(viewer.locator("#operator-list .op")).to_have_count(1)
    assert_selected(viewer, "Edge check", "+Z2 Z8", "Stabilizer generator")
    viewer.get_by_role("searchbox").fill("")
    expect(viewer.locator("#operator-list .op")).to_have_count(2)
    assert_selected(viewer, "Edge check", "+Z2 Z8", "Stabilizer generator")


def test_no_matches_clear_all_support_and_restore_results(viewer):
    expect(viewer.locator("#patch-diagram > polygon")).to_have_count(1)
    viewer.get_by_role("searchbox").fill("no such operator")
    expect(viewer.locator("#operator-list .op")).to_have_count(0)
    expect(viewer.locator("#operator-list .empty")).to_have_text("No matching operators.")
    expect(viewer.locator("#operator-name")).to_have_text("No matching operators")
    expect(viewer.locator("#figure-title")).to_have_text("Patch geometry")
    expect(viewer.locator("#category-label")).to_have_text("Geometry only")
    expect(viewer.locator("#pauli")).to_have_text("—")
    expect(viewer.locator("#weight")).to_be_hidden()
    expect(viewer.locator("#patch-diagram > polygon")).to_have_count(0)
    expect(viewer.locator('#patch-diagram > line[stroke-opacity]')).to_have_count(0)
    expect(viewer.locator("#patch-diagram .q text")).to_have_text(["2", "8", "13"])
    expect(viewer.locator('#patch-diagram .q > circle[fill="#fff"]')).to_have_count(3)
    expect(viewer.locator("#patch-diagram .q title")).to_have_text([
        "q2: (0, 0), data", "q8: (2, 0), data", "q13: (1, 1), data",
    ])
    viewer.get_by_role("searchbox").fill("")
    assert_selected(viewer, "Triangle check", "+X2 X8 X13", "Stabilizer generator")
    expect(viewer.locator("#patch-diagram > polygon")).to_have_count(1)


def test_patch_with_no_operators_has_a_distinct_empty_state(page, tmp_path):
    patch = BrowserPatch()
    patch.stabilizers.clear()
    patch.logical_ops.clear()
    path = patch.visualize().save_html(tmp_path / "empty.html")
    page.goto(path.as_uri())
    expect(page.locator("#operator-name")).to_have_text("No operators declared")
    expect(page.locator("#operator-list .empty")).to_have_text("No operators declared.")
    expect(page.locator("#detail-meta")).to_have_text(
        "The patch has no stabilizer, logical or gauge records."
    )
    expect(page.locator("#patch-diagram .q")).to_have_count(3)
    page.get_by_role("searchbox").fill("no such operator")
    expect(page.locator("#operator-name")).to_have_text("No operators declared")


def test_notebook_iframe_search_and_selection_are_independent(page, tmp_path):
    first = BrowserPatch().visualize(title="First patch")
    second = BrowserPatch().visualize(title="Second patch")
    path = tmp_path / "notebook.html"
    path.write_text(first._repr_html_() + second._repr_html_(), encoding="utf-8")
    page.goto(path.as_uri())
    first_frame = page.frame_locator("iframe").nth(0)
    second_frame = page.frame_locator("iframe").nth(1)
    expect(first_frame.locator("#title")).to_have_text("First patch")
    expect(second_frame.locator("#title")).to_have_text("Second patch")
    first_frame.get_by_role("button", name="Logicals 1", exact=True).click()
    first_frame.get_by_role("searchbox").fill("no such operator")
    expect(first_frame.locator("#operator-name")).to_have_text("No matching operators")
    assert_selected(second_frame, "Triangle check", "+X2 X8 X13", "Stabilizer generator")
    expect(second_frame.get_by_role("searchbox")).to_have_value("")
    second_frame.get_by_role("searchbox").fill("Edge")
    assert_selected(second_frame, "Edge check", "+Z2 Z8", "Stabilizer generator")
    expect(first_frame.locator("#operator-name")).to_have_text("No matching operators")
    first_frame.get_by_role("searchbox").fill("")
    assert_selected(first_frame, "Logical Y", "+Y8 Y13", "Logical representative")
