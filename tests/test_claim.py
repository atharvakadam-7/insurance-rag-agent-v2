import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from app.claim import calculate_claim


def test_waiting_period_blocks_claim():
    result = calculate_claim(claim_amount=50000, waiting_period_active=True)
    assert result.waiting_period_blocks_claim is True
    assert result.reimbursement == 0
    assert "waiting period" in result.as_text().lower()


def test_full_coverage_no_deductions():
    result = calculate_claim(claim_amount=50000, coverage_percent=100)
    assert result.reimbursement == 50000


def test_deductible_reduces_amount():
    result = calculate_claim(claim_amount=50000, deductible=5000, coverage_percent=100)
    assert result.amount_after_deductible == 45000
    assert result.reimbursement == 45000


def test_deductible_cannot_go_negative():
    result = calculate_claim(claim_amount=1000, deductible=5000, coverage_percent=100)
    assert result.amount_after_deductible == 0
    assert result.reimbursement == 0


def test_sublimit_caps_amount():
    result = calculate_claim(claim_amount=50000, sublimit=20000, coverage_percent=100)
    assert result.amount_after_sublimit == 20000
    assert result.reimbursement == 20000
    assert any("sub-limit" in n.lower() for n in result.notes)


def test_sublimit_no_effect_when_above_claim():
    result = calculate_claim(claim_amount=50000, sublimit=100000, coverage_percent=100)
    assert result.amount_after_sublimit == 50000
    assert result.notes == []


def test_copay_percent_reduces_reimbursement():
    result = calculate_claim(claim_amount=50000, coverage_percent=100, copay_percent=10)
    assert result.covered_amount == 50000
    assert result.copay_amount == 5000
    assert result.reimbursement == 45000


def test_coverage_percent_applied():
    result = calculate_claim(claim_amount=50000, coverage_percent=80)
    assert result.covered_amount == 40000
    assert result.reimbursement == 40000


def test_room_rent_proportionate_deduction():
    # claimed 8000/day vs cap 4000/day -> scale whole claim by 0.5
    result = calculate_claim(
        claim_amount=50000,
        coverage_percent=100,
        room_rent_cap=4000,
        room_rent_claimed=8000,
    )
    assert result.amount_after_room_rent == 25000
    assert result.reimbursement == 25000
    assert any("room rent" in n.lower() for n in result.notes)


def test_room_rent_no_deduction_when_under_cap():
    result = calculate_claim(
        claim_amount=50000,
        coverage_percent=100,
        room_rent_cap=8000,
        room_rent_claimed=4000,
    )
    assert result.amount_after_room_rent == 50000
    assert result.notes == []


def test_negative_claim_amount_clamped_to_zero():
    result = calculate_claim(claim_amount=-1000, coverage_percent=100)
    assert result.reimbursement == 0


def test_full_pipeline_order_room_rent_then_deductible_then_sublimit_then_coverage_then_copay():
    result = calculate_claim(
        claim_amount=100000,
        coverage_percent=90,
        deductible=10000,
        copay_percent=10,
        sublimit=70000,
        room_rent_cap=4000,
        room_rent_claimed=5000,  # ratio 0.8
    )
    assert result.amount_after_room_rent == 80000
    assert result.amount_after_deductible == 70000
    assert result.amount_after_sublimit == 70000  # sublimit == deductible amount, no further cap
    assert result.covered_amount == 63000  # 70000 * 0.9
    assert result.copay_amount == 6300  # 63000 * 0.10
    assert result.reimbursement == 56700


def test_as_text_format_normal_case():
    result = calculate_claim(claim_amount=50000, coverage_percent=100)
    text = result.as_text()
    assert "Estimated reimbursement: Rs 50,000.00" in text