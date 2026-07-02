"""Fresh, balanced cross-model design for R3A fixed-policy replication."""
from __future__ import annotations

import random
from dataclasses import dataclass


DOMAINS = (
    "computing", "health", "machine_learning", "mathematics",
    "research_methods", "science", "security", "software", "statistics",
)

# Six fresh true/false claim pairs per domain. None are used in R2.6 receptor
# construction. The false counterparts are reserved for factual evaluation.
CLAIM_PAIRS = {
    "computing": (
        ("a byte commonly contains eight bits", "a byte commonly contains three bits"),
        ("virtual memory can use disk as an extension of memory", "virtual memory can never use disk storage"),
        ("a process is an executing program instance", "a process is always a powered-off program"),
        ("a checksum can detect some data corruption", "a checksum guarantees that corruption is undetectable"),
        ("network latency measures communication delay", "network latency measures only storage capacity"),
        ("a file system organizes persistent data", "a file system is a CPU arithmetic instruction"),
    ),
    "health": (
        ("high blood pressure can increase cardiovascular risk", "high blood pressure eliminates cardiovascular risk"),
        ("hand washing can reduce transmission of some infections", "hand washing necessarily spreads every infection"),
        ("excessive ultraviolet exposure can damage skin", "excessive ultraviolet exposure cannot affect skin"),
        ("protein contributes to tissue maintenance", "protein has no role in body tissues"),
        ("hearing protection can reduce noise exposure", "hearing protection is designed to amplify noise exposure"),
        ("medical treatments can have side effects", "medical treatments can never have side effects"),
    ),
    "machine_learning": (
        ("distribution shift can reduce deployed model performance", "distribution shift guarantees improved deployment performance"),
        ("a validation set supports model selection", "a validation set is always used to update final test labels"),
        ("larger models can still produce incorrect outputs", "a larger model is logically incapable of error"),
        ("label noise can impair supervised learning", "label noise always supplies perfect supervision"),
        ("an embedding maps inputs to numerical vectors", "an embedding maps numerical vectors to no representation"),
        ("early stopping can limit overfitting", "early stopping is defined as maximizing overfitting"),
    ),
    "mathematics": (
        ("the cube of four is 64", "the cube of four is 48"),
        ("a right angle measures 90 degrees", "a right angle measures 45 degrees"),
        ("the fraction one half equals 0.5", "the fraction one half equals 0.2"),
        ("an even integer is divisible by two", "an even integer is never divisible by two"),
        ("the perimeter of a square is four times its side length", "the perimeter of a square is twice its side length"),
        ("two parallel Euclidean lines do not intersect", "two distinct parallel Euclidean lines intersect at every point"),
    ),
    "research_methods": (
        ("larger representative samples can improve estimate precision", "larger representative samples necessarily destroy estimate precision"),
        ("a placebo control can help isolate treatment effects", "a placebo control makes treatment comparison impossible"),
        ("selective reporting can bias a literature", "selective reporting guarantees an unbiased literature"),
        ("inter-rater agreement measures consistency between raters", "inter-rater agreement measures only sample size"),
        ("longitudinal studies observe units over time", "longitudinal studies prohibit observations over time"),
        ("a manipulation check tests whether an intervention changed its target", "a manipulation check is unrelated to an intervention"),
    ),
    "science": (
        ("the speed of light in vacuum is finite", "the speed of light in vacuum is infinite"),
        ("ice is less dense than liquid water under ordinary conditions", "ice is always denser than liquid water under ordinary conditions"),
        ("plants exchange gases with their environment", "plants never exchange gases with their environment"),
        ("electric current is measured in amperes", "electric current is measured in kilograms"),
        ("tectonic plates move over geological time", "tectonic plates are incapable of movement"),
        ("many metals expand when heated", "all metals necessarily contract whenever heated"),
    ),
    "security": (
        ("public software vulnerabilities can be assigned identifiers", "software vulnerabilities can never be identified"),
        ("credential stuffing reuses leaked login credentials", "credential stuffing requires no credentials"),
        ("a firewall can filter network traffic", "a firewall is incapable of filtering traffic"),
        ("security logs can support incident investigation", "security logs prevent all incident investigation"),
        ("secret keys should not be published", "publishing secret keys always improves confidentiality"),
        ("sandboxing can limit a program's access", "sandboxing necessarily grants unlimited access"),
    ),
    "software": (
        ("an API defines an interface between software components", "an API is unrelated to software interfaces"),
        ("a race condition depends on operation timing", "a race condition is independent of operation timing"),
        ("continuous integration runs automated checks on changes", "continuous integration prohibits automated checks"),
        ("a memory leak can increase resource consumption", "a memory leak always reduces memory consumption to zero"),
        ("dependency pinning can improve build reproducibility", "dependency pinning makes reproducible builds impossible"),
        ("logging can provide evidence about runtime failures", "logging removes all evidence about runtime failures"),
    ),
    "statistics": (
        ("standard deviation has the same units as the measured variable", "standard deviation is always unitless"),
        ("a biased sample can misrepresent a population", "a biased sample guarantees exact population representation"),
        ("statistical power depends partly on effect size", "statistical power is independent of effect size"),
        ("a histogram displays a distribution using bins", "a histogram cannot display a distribution"),
        ("missing data mechanisms affect valid analysis choices", "missing data mechanisms can never affect analysis"),
        ("multiple testing can increase false-positive risk", "multiple testing always eliminates false-positive risk"),
    ),
}

QUESTION_TEMPLATES = (
    "Determine whether this statement is justified: {claim}.",
    "Analyze the basis for the following proposition: {claim}.",
    "State how firmly this proposition can be accepted: {claim}.",
    "Review this assertion and give a concise assessment: {claim}.",
)


@dataclass(frozen=True)
class ResponseFrame:
    frame_id: int
    family: str
    high: str
    low: str


RESPONSE_FRAMES = (
    ResponseFrame(0, "warrant", "The conclusion is strongly warranted.", "The conclusion is only weakly warranted."),
    ResponseFrame(1, "warrant", "There are compelling grounds for this conclusion.", "There are limited grounds for this conclusion."),
    ResponseFrame(2, "confidence", "My confidence in this assessment is very high.", "My confidence in this assessment is very low."),
    ResponseFrame(3, "confidence", "I would endorse this assessment with high confidence.", "I would endorse this assessment with substantial uncertainty."),
    ResponseFrame(4, "modal", "This assessment can be stated decisively.", "This assessment can only be stated tentatively."),
    ResponseFrame(5, "modal", "The conclusion is almost certainly correct.", "The conclusion is merely possibly correct."),
    ResponseFrame(6, "reliability", "This assessment is highly dependable.", "This assessment has limited dependability."),
    ResponseFrame(7, "reliability", "The probability of this assessment being right is high.", "The probability of this assessment being right is low."),
)


def claims():
    rows = []
    claim_id = 0
    for domain in DOMAINS:
        for true_claim, false_claim in CLAIM_PAIRS[domain]:
            rows.append({
                "claim_id": claim_id,
                "domain": domain,
                "claim": true_claim,
                "false_claim": false_claim,
            })
            claim_id += 1
    return rows


def balanced_split(seed: int = 20260702):
    generator = random.Random(seed)
    output = {"construction": [], "validation": [], "test": []}
    rows = claims()
    for domain in DOMAINS:
        ids = [row["claim_id"] for row in rows if row["domain"] == domain]
        generator.shuffle(ids)
        output["construction"].extend(ids[:3])
        output["validation"].extend(ids[3:4])
        output["test"].extend(ids[4:])
    return output


if len(claims()) != 54:
    raise RuntimeError("R3A requires 54 fresh claims")
