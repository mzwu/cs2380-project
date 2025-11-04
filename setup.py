from pabutools.election import parse_pabulib, Cost_Sat
from pabutools.rules import method_of_equal_shares
instance, profile = parse_pabulib("data/netherlands_amsterdam_252_.pb")
winners = method_of_equal_shares(
    instance,
    profile.as_multiprofile(),
    sat_class=Cost_Sat,
    voter_budget_increment=1 # use the completion method Add1
)
print(winners)