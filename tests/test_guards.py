from app.guards import money_values, prices_grounded, valid_sources
from app.ingest import chunk_blocks, extract_blocks

CTX = "Package price: starting from $5,500 USD. Includes lodging. Consultation 150 USD."


def test_price_in_context_passes():
    assert prices_grounded("It starts from $5,500 USD.", CTX)
    assert prices_grounded("Consultation is 150 USD.", CTX)


def test_invented_price_fails():
    assert not prices_grounded("It costs $50.", CTX)
    assert not prices_grounded("About 7,000 dollars.", CTX)


def test_no_price_is_fine():
    assert prices_grounded("The package includes lodging.", CTX)


def test_money_values_parse():
    assert money_values("US$1,200.50 and 300 EUR") == {1200.5, 300.0}


def test_sources_must_be_retrieved():
    assert valid_sources(["a", "evil", "a", "b"], {"a", "b"}) == ["a", "b"]


def test_table_kept_intact_and_headings_attached():
    html = "<main><h1>Title</h1><h2>Prices</h2><table><tr><th>Item</th><th>USD</th></tr><tr><td>Stay</td><td>$100</td></tr></table></main>"
    title, blocks = extract_blocks(html)
    chunks = chunk_blocks(blocks + [("b", "x" * 60)])
    assert title == "Title"
    assert any("Item | USD\nStay | $100" in c.text for c in chunks)
    assert chunks[0].text.startswith("## Title")
