from src.retrievers.rules import rule_based_filter
from src.models.product_doc import ProductDoc


def test_rule_based_filter_basic():
    docs = [
        ProductDoc(
            id=0,
            raw={"product_info": {"name": "TWIN BOX", "category": "Box TV"}},
            text="TWIN BOX box tv",
            keywords=["twin box", "android tv"],
            product_name="twin box",
            category="box tv",
            provider="algérie télécom",
        ),
        ProductDoc(
            id=1,
            raw={"product_info": {"name": "MOALIM", "category": "Éducation"}},
            text="MOALIM soutien scolaire",
            keywords=["moalim", "soutien scolaire"],
            product_name="moalim",
            category="éducation",
            provider="algérie télécom",
        ),
    ]

    q = "Quel est le prix de la TWIN BOX ?"
    cands = rule_based_filter(q, docs)
    assert len(cands) == 1
    assert cands[0].product_name == "twin box"
