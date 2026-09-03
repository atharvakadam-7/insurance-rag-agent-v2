from app.claim import calculate_claim


def test_basic_percent():
    result = calculate_claim(100_000, coverage_percent=80, deductible=0)
    assert result.reimbursement == 80_000


def test_deductible_and_copay():
    result = calculate_claim(50_000, coverage_percent=100, deductible=5_000, copay_percent=20)
    assert result.amount_after_deductible == 45_000
    assert result.copay_amount == 9_000
    assert result.reimbursement == 36_000


def test_sublimit():
    result = calculate_claim(80_000, sublimit=25_000)
    assert result.reimbursement == 25_000
    assert any("Sub-limit" in n for n in result.notes)


def test_waiting_period_zero():
    result = calculate_claim(50_000, waiting_period_active=True)
    assert result.reimbursement == 0
    assert result.waiting_period_blocks_claim


def test_room_rent_proportionate():
    result = calculate_claim(
        40_000,
        room_rent_cap=5_000,
        room_rent_claimed=10_000,
    )
    assert result.reimbursement == 20_000
