from modules_hidrosed.granulometria_avanzada import (
    DEFAULT_STANDARD_PROFILE_KEY,
    build_default_grain_session_payload,
    get_standard_grain_profile,
    get_standard_grain_stats,
    recommend_standard_profile_by_context,
)

payload = build_default_grain_session_payload(DEFAULT_STANDARD_PROFILE_KEY)
assert payload['grain_default_profile_active'] is True
assert len(payload['grain_df']) >= 5
assert len(payload['grain_stats_multi']) == 1
stats = get_standard_grain_stats(DEFAULT_STANDARD_PROFILE_KEY)
for col in ['D10_mm','D50_mm','D84_mm','D90_mm','Fuente_granulometria']:
    assert col in stats.columns, col
assert float(stats.loc[0, 'D50_mm']) > 0
assert recommend_standard_profile_by_context(slope=0.02) in ['aluvial_semiarido_mixto','grava_gruesa_bolones']
curve = get_standard_grain_profile('arena_media')
assert curve['porcentaje_pasa'].is_monotonic_increasing
print('OK v6.2 default granulometry profiles')
