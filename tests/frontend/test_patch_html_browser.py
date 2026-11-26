"""Optional real-browser checks; install Playwright and Chromium to run them.

    python -m pip install playwright
    python -m playwright install chromium
    python -m pytest tests/frontend/test_patch_html_browser.py

Neither Playwright nor its browser is required by the core test suite.
"""

import pytest

from lightstim.ir.qec_patch import QECPatch
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCode


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
    expect(page.locator('.layer[data-layer="all"]')).to_have_attribute("aria-pressed", "true")
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
    expect(viewer.locator(".op-shape polygon")).to_have_count(1)
    viewer.get_by_role("searchbox").fill("no such operator")
    expect(viewer.locator("#operator-list .op")).to_have_count(0)
    expect(viewer.locator("#operator-list .empty")).to_have_text("No matching operators.")
    expect(viewer.locator("#operator-name")).to_have_text("No matching operators")
    expect(viewer.locator("#figure-title")).to_have_text("Patch geometry")
    expect(viewer.locator("#category-label")).to_have_text("Geometry only")
    expect(viewer.locator("#pauli")).to_have_text("—")
    expect(viewer.locator("#weight")).to_be_hidden()
    expect(viewer.locator(".op-shape")).to_have_count(0)
    expect(viewer.locator(".q .factor-label")).to_have_count(0)
    viewer.get_by_label("Qubit IDs").check()
    expect(viewer.locator(".qubit-label")).to_have_text(["2", "8", "13"])
    expect(viewer.locator("#patch-diagram .q title")).to_have_text([
        "q2: (0, 0), data", "q8: (2, 0), data", "q13: (1, 1), data",
    ])
    viewer.get_by_role("searchbox").fill("")
    assert_selected(viewer, "Triangle check", "+X2 X8 X13", "Stabilizer generator")
    expect(viewer.locator(".op-shape polygon")).to_have_count(1)


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
    expect(second_frame.locator(".op.selected")).to_have_count(0)
    expect(second_frame.locator('.op-shape[data-state="overview"]')).to_have_count(2)
    expect(second_frame.get_by_role("searchbox")).to_have_value("")
    second_frame.get_by_role("searchbox").fill("Edge")
    assert_selected(second_frame, "Edge check", "+Z2 Z8", "Stabilizer generator")
    expect(first_frame.locator("#operator-name")).to_have_text("No matching operators")
    first_frame.get_by_role("searchbox").fill("")
    assert_selected(first_frame, "Logical Y", "+Y8 Y13", "Logical representative")


def test_overview_layers_and_operator_focus(viewer):
    overview = viewer.locator('.op-shape[data-state="overview"]')
    expect(overview).to_have_count(2)
    expect(viewer.locator(".op.selected")).to_have_count(0)
    viewer.get_by_role("button", name="X Stabilizers", exact=True).click()
    expect(overview).to_have_count(1)
    expect(overview).to_have_attribute("data-basis", "X")
    overview.locator("polygon").click()
    expect(viewer.locator('.op-shape[data-state="focus"]')).to_have_count(1)
    edge = viewer.locator('.op-shape[data-state="context"] line')
    # The Z check lies on an edge of the selected triangle. Click its midpoint
    # in screen coordinates so the browser performs real SVG hit testing.
    x, y = edge.evaluate("""line => {
        const p = new DOMPoint((line.x1.baseVal.value + line.x2.baseVal.value) / 2,
                               (line.y1.baseVal.value + line.y2.baseVal.value) / 2);
        const screen = p.matrixTransform(line.getScreenCTM());
        return [screen.x, screen.y];
    }""")
    viewer.mouse.click(x, y)
    assert_selected(viewer, "Edge check", "+Z2 Z8", "Stabilizer generator")
    viewer.locator("#back-to-layer").click()
    expect(overview).to_have_count(1)
    expect(overview).to_have_attribute("data-basis", "X")
    expect(viewer.locator(".op.selected")).to_have_count(0)


def test_system_membership_and_global_qubit_details(page, tmp_path):
    from lightstim.ir.qec_system import QECSystem
    from lightstim.qec_code.surface_code.rotated import RotatedTwoPatchCoupler

    system = QECSystem()
    system.add_patch(RotatedSurfaceCode(distance=3), name="left")
    system.add_patch(RotatedSurfaceCode(distance=3), name="right", offset=(8, 0))
    system.register_coupler(RotatedTwoPatchCoupler(), ["left", "right"], "seam", interaction_type="XX")
    output = system.visualize().save_html(tmp_path / "system.html")
    page.goto(output.as_uri())
    expect(page.locator('.op-shape[data-state="overview"]')).to_have_count(len(system.active_stabilizers))
    expect(page.get_by_role("button", name="Patch Boundaries", exact=True)).to_have_count(0)
    seam = page.get_by_role("button", name="seam", exact=True)
    seam.click()
    expect(seam).to_have_attribute("aria-pressed", "true")
    expect(page.locator(".patch-region")).to_have_count(0)
    q = min(system.local_to_global_map["right"].values())
    page.locator(f".q[data-qubit-id='{q}']").press("Enter")
    expect(page.locator("#qubit-info")).to_contain_text(f"q{q}")
    expect(page.locator("#qubit-info")).to_contain_text("right")
    page.get_by_role("searchbox").fill("seam")
    expect(page.locator("#detail-meta")).to_contain_text("inactive")


def test_encoded_logical_layers_preserve_mixed_physical_factors(page, tmp_path):
    from lightstim.qec_code.surface_code.xzzx import XZZXSurfaceCode

    patch = XZZXSurfaceCode(distance=3)
    output = patch.visualize().save_html(tmp_path / "xzzx.html")
    page.goto(output.as_uri())
    for axis in ("X", "Z"):
        page.get_by_role("button", name=f"Logical {axis}", exact=True).click()
        expect(page.locator('.op-shape[data-state="overview"][data-category="logical"]')).to_have_count(1)
        operator = page.locator("#operator-list .op")
        expect(operator).to_have_count(1)
        operator.click()
        record = next(op for op in patch.logical_ops if op["type"] == axis)
        for q, factor in record["pauli"].items():
            expect(page.locator(f'.q[data-qubit-id="{q}"] .factor-label')).to_have_text(factor)
        expect(page.locator(".q:has(.logical-crossing)")).to_have_count(1)


@pytest.mark.parametrize("factory,weight", [(RotatedSurfaceCode, 2), (UnrotatedSurfaceCode, 3)])
def test_triangular_checks_and_neutral_syndrome_circles(page, tmp_path, factory, weight):
    patch = factory(distance=3)
    page.goto(patch.visualize().save_html(tmp_path / "geometry.html").as_uri())
    checks = page.locator('.q[data-role="check"]')
    expect(page.locator("#legend-aux")).to_have_text("Syndrome")
    expect(checks.locator("circle.glyph")).to_have_count(len(patch.syndrome_indices))
    styles = checks.locator(".glyph").evaluate_all("""nodes => nodes.map(n => {
        const style = getComputedStyle(n);
        return [style.fill, style.stroke];
    })""")
    assert len({tuple(style) for style in styles}) == 1
    assert styles[0][0] != styles[0][1]
    for i, record in enumerate(patch.stabilizers):
        if len(record["pauli"]) != weight:
            continue
        # At weight two only the extra geometric vertex belongs to the syndrome.
        vertices = set(record["pauli"])
        if weight == 2:
            vertices.add(record["syn_idx"])
        shape = page.locator(f'.op-shape[data-operator-id="stabilizer-{i}"] polygon')
        actual = shape.evaluate("p => Array.from(p.points, v => [v.x, v.y])")
        expected = [
            page.locator(f'.q[data-qubit-id="{q}"] .glyph')
            .evaluate("c => [c.cx.baseVal.value, c.cy.baseVal.value]")
            for q in vertices
        ]
        assert sorted(actual) == sorted(expected)


def test_switch_checks_from_a_logical_or_another_z_check(page, tmp_path):
    patch = RotatedSurfaceCode(distance=3)
    page.goto(patch.visualize().save_html(tmp_path / "inspect.html").as_uri())
    logical_z = page.get_by_role("button", name="Logical Z", exact=True)
    logical_z.click()
    page.locator("#operator-list .op").first.click()
    for i, record in enumerate(patch.stabilizers):
        if record["type"] != "Z":
            continue
        # Aim at the hollow marker in the centre of the check.
        site = record["syn_idx"]
        page.locator(f'.q[data-qubit-id="{site}"] .glyph').click()
        expect(page.locator('.op-shape[data-state="focus"]')).to_have_attribute("data-operator-id", f"stabilizer-{i}")
        expect(page.locator("#qubit-info")).to_contain_text(f"q{site}")
    page.keyboard.press("Escape")
    expect(logical_z).to_have_attribute("aria-pressed", "true")
    expect(page.locator(".op.selected")).to_have_count(0)


def test_shared_syndrome_qubit_preserves_operator_selection(page, tmp_path):
    patch = BrowserPatch()
    site = patch.add_qubit(3, 1, "syndrome")
    for check in patch.stabilizers:
        check["syn_idx"] = site
    page.goto(patch.visualize().save_html(tmp_path / "shared.html").as_uri())
    page.get_by_role("button", name="Logicals 1", exact=True).click()
    page.locator(f'.q[data-qubit-id="{site}"]').press("Enter")
    expect(page.locator("#operator-name")).to_have_text("Logical Y")
    expect(page.locator("#qubit-info")).to_contain_text(f"q{site}")
