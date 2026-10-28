import pytest

from sentinel.rules import RuleEngine


@pytest.fixture
def engine(config):
    return RuleEngine(config.rules)


def acting(engine, msg, history=None):
    return {(h.rule, h.action) for h in engine.evaluate(msg, history) if h.action != "ignore"}


@pytest.mark.parametrize(
    "text, rule",
    [
        ("mi cel es 264 4123456", "phone"),
        ("agendame: +54 9 11 5555-1234", "phone"),
        ("escribime a juan.perez@gmail.com", "email"),
        ("mandame un mail a juanperez arroba hotmail punto com", "email"),
        ("seguime en ig: @tienda.mariposa", "social_handle"),
        ("pasame tu número así te escribo por wsp", "contact_request"),
        ("entrá a www.plata-facil.xyz ya", "external_link"),
        ("mirá bit.ly/3xYz9Q", "external_link"),
        ("GANANCIAS DIARIAS para vos", "spam_phrases"),
        ("sos un h1j0 d3 put4", "strong_insults"),
        ("te voy a cagar a trompadas", "threats"),
        ("estos bolitas de mierda", "slurs"),
        ("sos un mogólico", "slurs"),
    ],
)
def test_rule_catches_harmful_text(engine, message, text, rule):
    assert rule in {name for name, _ in acting(engine, message(text))}


@pytest.mark.parametrize(
    "text",
    [
        "facturaría unos $ 15.000.000 al año",
        "el turno es el 28/11/2026",
        "el plan 2026-2027 está listo",
        "te paso el link https://meet.google.com/abc-defg-hij",
        "compartilo por https://docs.google.com/document/d/1AbC",
        "jajaja boludo sos un genio",
        "el jueves a las 18 me queda perfecto",
    ],
)
def test_rule_ignores_normal_text(engine, message, text):
    assert acting(engine, message(text)) == set()


def test_mild_insult_is_recorded_but_not_acted_on(engine, message):
    hits = engine.evaluate(message("che boludo, ya llegué"))
    assert [(h.rule, h.action) for h in hits] == [("mild_insults", "ignore")]


def test_repeated_message_flags_on_third_copy(engine, message):
    text = "Hola! tengo una propuesta de negocio increíble"

    class History:
        def __init__(self, previous):
            self.previous = previous

        def recent_texts_by_author(self, msg, since):
            return self.previous

    assert acting(engine, message(text), History([text])) == set()
    assert ("repeated_message", "flag") in acting(engine, message(text), History([text, text]))
    assert acting(engine, message("dale"), History(["dale", "dale"])) == set()
