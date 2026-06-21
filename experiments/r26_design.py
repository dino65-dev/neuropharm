"""Balanced R2.6 claim, question, response-frame, and certainty design."""
from __future__ import annotations

import random
from dataclasses import dataclass


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DOMAINS = (
    "computing",
    "health",
    "machine_learning",
    "mathematics",
    "research_methods",
    "science",
    "security",
    "software",
    "statistics",
)

# Each item is (true statement, matched false statement). Construction uses the
# true statement. The false counterpart is reserved for fresh factual endpoints.
CLAIM_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "computing": (
        ("RAM is volatile working memory", "RAM retains all data without power"),
        ("a compiler translates source code", "a compiler is a physical storage device"),
        ("caching can reduce repeated computation", "caching always increases repeated computation"),
        ("binary digits take values zero or one", "a binary digit has ten possible values"),
        ("a CPU executes machine instructions", "a CPU is used only for permanent file storage"),
        ("lossless compression permits exact reconstruction", "lossless compression necessarily destroys information"),
        ("an operating system manages hardware resources", "an operating system has no role in resource management"),
        ("Unicode represents text using coded characters", "Unicode is an image-only compression format"),
    ),
    "health": (
        ("adequate hydration supports normal physiology", "the human body never requires water"),
        ("regular exercise generally benefits cardiovascular health", "regular exercise necessarily eliminates cardiovascular fitness"),
        ("sleep supports memory consolidation", "sleep prevents memory consolidation"),
        ("sunscreen can reduce ultraviolet exposure", "sunscreen increases ultraviolet exposure by design"),
        ("vaccination can reduce infectious-disease spread", "vaccination cannot affect infectious-disease spread"),
        ("smoking increases the risk of several diseases", "smoking eliminates the risk of several diseases"),
        ("dietary fiber supports digestive health", "dietary fiber has no interaction with digestion"),
        ("antibiotics do not treat viral infections directly", "antibiotics directly kill every virus"),
    ),
    "machine_learning": (
        ("data leakage can inflate evaluation performance", "data leakage guarantees an unbiased evaluation"),
        ("regularization can reduce overfitting", "regularization is defined as maximizing overfitting"),
        ("cross-validation estimates out-of-sample performance", "cross-validation uses no held-out observations"),
        ("gradient clipping can limit optimization instability", "gradient clipping always makes gradients unbounded"),
        ("quantization can alter model behavior", "quantization can never alter model behavior"),
        ("tokenization affects model input length", "tokenization cannot affect model input length"),
        ("training loss alone does not guarantee deployment quality", "training loss alone proves perfect deployment quality"),
        ("class imbalance can distort accuracy-based evaluation", "class imbalance can never affect accuracy-based evaluation"),
    ),
    "mathematics": (
        ("twelve multiplied by thirteen equals 156", "twelve multiplied by thirteen equals 145"),
        ("the square root of 144 is 12", "the square root of 144 is 14"),
        ("matrix multiplication is generally not commutative", "matrix multiplication is always commutative"),
        ("a derivative describes a local rate of change", "a derivative is unrelated to rates of change"),
        ("the angles of a Euclidean triangle sum to 180 degrees", "the angles of a Euclidean triangle sum to 270 degrees"),
        ("a prime number has exactly two positive divisors", "a prime number has exactly four positive divisors"),
        ("the mean is sensitive to extreme outliers", "the arithmetic mean is invariant to every outlier"),
        ("the probability of a certain event is one", "the probability of a certain event is zero"),
    ),
    "research_methods": (
        ("random assignment reduces treatment-selection bias", "random assignment intentionally maximizes treatment-selection bias"),
        ("control groups support causal inference", "control groups make causal comparisons impossible"),
        ("independent replication strengthens empirical confidence", "independent replication provides no empirical information"),
        ("blinding can reduce observer bias", "blinding is designed to increase observer bias"),
        ("negative controls can expose experimental confounds", "negative controls cannot reveal experimental confounds"),
        ("pre-registration can reduce analytic flexibility", "pre-registration is performed only after all analyses are finalized"),
        ("measurement error can bias estimated effects", "measurement error can never bias an estimated effect"),
        ("correlation alone does not establish causation", "correlation alone proves causation without assumptions"),
    ),
    "science": (
        ("Earth's axial tilt causes the seasons", "Earth's daily rotation alone causes the seasons"),
        ("photosynthesis converts light energy into chemical energy", "photosynthesis converts chemical energy into no stored energy"),
        ("water boils near 100 degrees Celsius at sea level", "water boils near zero degrees Celsius at sea level"),
        ("the Moon can block the Sun during a solar eclipse", "Earth always blocks the Sun during a solar eclipse"),
        ("gravity accelerates unsupported objects toward Earth", "gravity accelerates unsupported objects away from Earth"),
        ("atoms contain electrons and a nucleus", "atoms contain no subatomic structure"),
        ("DNA carries hereditary information", "DNA contains no hereditary information"),
        ("sound requires a medium in ordinary acoustic propagation", "ordinary acoustic sound propagates through a perfect vacuum"),
    ),
    "security": (
        ("passwords should be unique across websites", "reusing one password everywhere reduces credential risk"),
        ("least-privilege access reduces security exposure", "least-privilege access grants every user every permission"),
        ("encryption protects data confidentiality", "encryption is designed to publish plaintext"),
        ("input validation reduces malformed-data failures", "input validation requires accepting every malformed input"),
        ("rate limiting can reduce service abuse", "rate limiting necessarily increases unlimited service abuse"),
        ("multi-factor authentication can reduce account-takeover risk", "multi-factor authentication removes all authentication factors"),
        ("software patches can remediate known vulnerabilities", "software patches cannot remediate known vulnerabilities"),
        ("phishing attempts often impersonate trusted entities", "phishing never impersonates a trusted entity"),
    ),
    "software": (
        ("automated tests improve regression detection", "automated tests make regression detection impossible"),
        ("version control preserves a history of changes", "version control necessarily deletes all change history"),
        ("database indexes can accelerate queries", "database indexes can never accelerate a query"),
        ("backups reduce data-loss risk", "backups necessarily increase permanent data loss"),
        ("gradual deployment can limit operational risk", "gradual deployment always maximizes simultaneous operational risk"),
        ("monitoring helps detect production failures", "monitoring prevents detection of production failures"),
        ("idempotent operations tolerate safe repetition", "an idempotent operation must change result on every repetition"),
        ("code review can identify defects before release", "code review cannot identify any defect before release"),
    ),
    "statistics": (
        ("sample size influences statistical precision", "sample size has no relationship to statistical precision"),
        ("confidence calibration differs from classification accuracy", "confidence calibration is identical to classification accuracy by definition"),
        ("outliers can strongly influence a sample mean", "outliers can never influence a sample mean"),
        ("the median is robust to extreme values", "the median is always dominated by one extreme value"),
        ("a confidence interval reflects sampling uncertainty", "a confidence interval contains no information about sampling uncertainty"),
        ("selection bias can distort population estimates", "selection bias guarantees representative population estimates"),
        ("variance measures dispersion around a mean", "variance is unrelated to dispersion"),
        ("a p-value is not the probability that the null hypothesis is true", "a p-value is exactly the probability that the null hypothesis is true"),
    ),
}

QUESTION_TEMPLATES = (
    "Assess whether {claim}.",
    "What evidence bears on the statement that {claim}?",
    "Give a concise judgment on this claim: {claim}.",
    "Evaluate the following statement: {claim}.",
)


@dataclass(frozen=True)
class ResponseFrame:
    frame_id: int
    family: str
    position: str
    high: str
    low: str


RESPONSE_FRAMES = (
    ResponseFrame(0, "evidence", "initial", "Strong evidence supports this conclusion.", "Weak evidence supports this conclusion."),
    ResponseFrame(1, "evidence", "final", "This conclusion is supported by strong evidence.", "This conclusion is supported by weak evidence."),
    ResponseFrame(2, "self_confidence", "initial", "I am highly confident that this conclusion is correct.", "I am highly uncertain whether this conclusion is correct."),
    ResponseFrame(3, "self_confidence", "final", "This conclusion is correct, and my confidence is high.", "This conclusion may be incorrect, and my confidence is low."),
    ResponseFrame(4, "modal", "initial", "Certainly, this conclusion is correct.", "Possibly, this conclusion is correct."),
    ResponseFrame(5, "modal", "final", "This conclusion is correct with certainty.", "This conclusion is correct only tentatively."),
    ResponseFrame(6, "reliability", "initial", "This conclusion is highly reliable.", "This conclusion has low reliability."),
    ResponseFrame(7, "reliability", "final", "I assign a high probability to this conclusion.", "I assign a low probability to this conclusion."),
)


def claims() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    claim_id = 0
    for domain in DOMAINS:
        pairs = CLAIM_PAIRS[domain]
        if len(pairs) != 8:
            raise ValueError(f"{domain} must contain exactly eight claim pairs")
        for within_domain_id, (true_claim, false_claim) in enumerate(pairs):
            rows.append({
                "claim_id": claim_id,
                "domain": domain,
                "within_domain_id": within_domain_id,
                "claim": true_claim,
                "false_claim": false_claim,
            })
            claim_id += 1
    return rows


def balanced_claim_split(seed: int) -> dict[str, list[int]]:
    """Return 4/2/2 claims per domain for construction/validation/test."""
    split = {"construction": [], "validation": [], "test": []}
    generator = random.Random(seed)
    by_domain: dict[str, list[int]] = {domain: [] for domain in DOMAINS}
    for row in claims():
        by_domain[str(row["domain"])].append(int(row["claim_id"]))
    for domain in DOMAINS:
        ids = by_domain[domain][:]
        generator.shuffle(ids)
        split["construction"].extend(ids[:4])
        split["validation"].extend(ids[4:6])
        split["test"].extend(ids[6:8])
    return split


def validate_design() -> None:
    rows = claims()
    if len(rows) != 72:
        raise ValueError("R2.6 requires exactly 72 claims")
    if len(QUESTION_TEMPLATES) != 4 or len(RESPONSE_FRAMES) != 8:
        raise ValueError("R2.6 requires four questions and eight response frames")
    if {frame.family for frame in RESPONSE_FRAMES} != {
        "evidence", "self_confidence", "modal", "reliability"
    }:
        raise ValueError("response-family design is incomplete")
    if {frame.position for frame in RESPONSE_FRAMES} != {"initial", "final"}:
        raise ValueError("certainty phrase position is not counterbalanced")


validate_design()
