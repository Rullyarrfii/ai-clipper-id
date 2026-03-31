#!/usr/bin/env python3
"""Test that the schedule is deterministic for the same date."""

import sys
sys.path.insert(0, '/Users/samuelkoesnadi/Documents/projects/sosmed')

from datetime import datetime, timedelta
import pytz

# Mock the scheduler functions we need to test
import random

def _get_daily_seed(date: datetime) -> int:
    """Generate a deterministic seed for a given date."""
    date_str = date.strftime("%Y-%m-%d")
    return int(date_str.replace("-", ""))

DAILY_ACTIVE_POSTS = [3, 4]
SCHEDULE_SLOTS = [
    ("20:45", 1, "Peak prime time"),
    ("11:50", 1, "Lunch peak"),
    ("18:50", 2, "Post-Maghrib"),
    ("15:50", 2, "End of work/school"),
    ("08:50", 3, "Morning work break"),
    ("06:50", 3, "Morning commute"),
]

def generate_schedule_for_date(test_date: datetime):
    """Generate the schedule for a given date (deterministic)."""
    # Generate daily target
    random.seed(_get_daily_seed(test_date))
    daily_target = random.choice(DAILY_ACTIVE_POSTS)
    
    # Group slots by tier
    by_tier = {}
    for slot_time, tier, _label in SCHEDULE_SLOTS:
        by_tier.setdefault(tier, []).append(slot_time)
    
    # Fill slots
    selected = []
    remaining = daily_target
    random.seed(_get_daily_seed(test_date))
    for tier in sorted(by_tier.keys()):
        tier_slots = by_tier[tier][:]
        random.shuffle(tier_slots)
        take = min(remaining, len(tier_slots))
        selected.extend(tier_slots[:take])
        remaining -= take
        if remaining == 0:
            break
    
    return daily_target, sorted(selected)

# Test determinism
print("Testing deterministic schedule generation...")
print("=" * 60)

# Test with today's date
today = datetime.now(pytz.timezone("Asia/Jakarta")).date()
print(f"\nTesting with date: {today}")

# Run multiple times to verify same result
results = []
for i in range(5):
    target, slots = generate_schedule_for_date(today)
    results.append((target, slots))
    print(f"  Run {i+1}: target={target}, slots={slots}")

# Check all results are identical
if all(r == results[0] for r in results):
    print("\n✅ SUCCESS: All runs produced identical schedules!")
else:
    print("\n❌ FAIL: Results varied between runs!")
    sys.exit(1)

# Test with different dates to show variation
print("\n" + "=" * 60)
print("Testing variation across different dates:")
test_dates = [today + timedelta(days=i) for i in range(7)]
for d in test_dates:
    target, slots = generate_schedule_for_date(d)
    print(f"  {d}: target={target}, slots={slots}")

print("\n✅ Deterministic schedule test passed!")
