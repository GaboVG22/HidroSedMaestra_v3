from modules_hidrosed.modulo_integracion_secciones_hidrosed import (
    generar_datos_demo_v13,
    validar_secciones_para_hidrosed,
    convertir_secciones_a_hidrosed,
    convertir_puntos_a_hidrosed,
)
from modules_hidrosed.modelacion_avanzada import (
    ConnectedProfileParams,
    compute_connected_profile_v6,
    monte_carlo_uncertainty_v6,
    confidence_score_v6,
    calibrate_manning_multiplier,
    water_density_kgm3,
)
import pandas as pd


def main():
    data = generar_datos_demo_v13(n_secciones=4, puntos_por_seccion=11)
    errores, advertencias, df_validado = validar_secciones_para_hidrosed(data['df_secciones_validas'], data['df_puntos_secciones'])
    assert not errores, errores
    h03 = convertir_secciones_a_hidrosed(df_validado, caudal_default=80.0, manning_default=0.035)
    h04 = convertir_puntos_a_hidrosed(data['df_puntos_secciones'], df_validado)
    p = ConnectedProfileParams(rho_w=water_density_kgm3(20), rho_s=2650, theta_c=0.047, temp_c=20)
    res = compute_connected_profile_v6(h03, h04, params=p, return_period_years=100)
    assert len(res) == len(h03), (len(res), len(h03))
    required = ['Cota_lamina_agua_m', 'Metodo_perfil', 'Perfil_conectado_converge', 'Delta_conectado_vs_normal_m', 'Gs_MPM_m3s', 'Socavacion_ajustada_m']
    for c in required:
        assert c in res.columns, c
    unc = monte_carlo_uncertainty_v6(h03, h04, params=p, n_runs=6, q_cv=0.05, n_cv=0.05, d50_cv=0.05, slope_cv=0.05)
    assert not unc.empty
    # Crear observaciones sintéticas para probar calibración. No implica calibración real.
    obs = res[['ID_Seccion', 'Cota_lamina_agua_m']].copy().rename(columns={'Cota_lamina_agua_m': 'Cota_lamina_observada_m'})
    factor, cal = calibrate_manning_multiplier(h03, h04, obs, params=p, multipliers=[0.95, 1.0, 1.05])
    assert 0.95 <= factor <= 1.05
    score, score_df = confidence_score_v6(res, uncertainty=unc, calibration=cal)
    assert score >= 8.0, score
    assert not score_df.empty
    print('OK modelacion v6', len(res), 'secciones', 'score', score)

if __name__ == '__main__':
    main()
