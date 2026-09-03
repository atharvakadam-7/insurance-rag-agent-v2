from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClaimBreakdown:
    waiting_period_blocks_claim: bool
    amount_after_room_rent: float
    amount_after_deductible: float
    amount_after_sublimit: float
    covered_amount: float
    copay_amount: float
    reimbursement: float
    notes: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        if self.waiting_period_blocks_claim:
            return (
                "Estimated reimbursement: Rs 0.00\n"
                "Reason: a waiting period still applies, so this claim is not payable "
                "under the supplied parameters."
            )
        lines = [
            f"Amount after room-rent adjustment: Rs {self.amount_after_room_rent:,.2f}",
            f"After deductible: Rs {self.amount_after_deductible:,.2f}",
            f"After sub-limit: Rs {self.amount_after_sublimit:,.2f}",
            f"After coverage %: Rs {self.covered_amount:,.2f}",
            f"Co-pay deducted: Rs {self.copay_amount:,.2f}",
            f"Estimated reimbursement: Rs {self.reimbursement:,.2f}",
        ]
        if self.notes:
            lines.append("Notes: " + "; ".join(self.notes))
        return "\n".join(lines)


def calculate_claim(
    claim_amount: float,
    coverage_percent: float = 100.0,
    deductible: float = 0.0,
    copay_percent: float = 0.0,
    sublimit: float | None = None,
    room_rent_cap: float | None = None,
    room_rent_claimed: float | None = None,
    waiting_period_active: bool = False,
) -> ClaimBreakdown:
    notes: list[str] = []
    if waiting_period_active:
        return ClaimBreakdown(
            waiting_period_blocks_claim=True,
            amount_after_room_rent=0,
            amount_after_deductible=0,
            amount_after_sublimit=0,
            covered_amount=0,
            copay_amount=0,
            reimbursement=0,
            notes=["Waiting period is still active for this condition."],
        )

    amount = max(float(claim_amount), 0.0)
    if (
        room_rent_cap is not None
        and room_rent_claimed is not None
        and room_rent_claimed > room_rent_cap
        and room_rent_claimed > 0
    ):
        ratio = room_rent_cap / room_rent_claimed
        reduced = amount * ratio
        notes.append(
            f"Room rent Rs {room_rent_claimed:,.2f} exceeds cap Rs {room_rent_cap:,.2f}; "
            f"claim scaled by {ratio:.2%}."
        )
        amount = reduced

    after_room = amount
    after_deductible = max(after_room - max(float(deductible), 0.0), 0.0)
    after_sublimit = after_deductible
    if sublimit is not None:
        after_sublimit = min(after_deductible, max(float(sublimit), 0.0))
        if after_sublimit < after_deductible:
            notes.append(f"Sub-limit of Rs {float(sublimit):,.2f} applied.")

    covered = after_sublimit * (float(coverage_percent) / 100.0)
    copay = covered * (max(float(copay_percent), 0.0) / 100.0)
    reimbursement = max(covered - copay, 0.0)
    return ClaimBreakdown(
        waiting_period_blocks_claim=False,
        amount_after_room_rent=after_room,
        amount_after_deductible=after_deductible,
        amount_after_sublimit=after_sublimit,
        covered_amount=covered,
        copay_amount=copay,
        reimbursement=reimbursement,
        notes=notes,
    )
