from modules_hidrosed.modulo_integracion_secciones_hidrosed import (
    generar_datos_demo_v13,
    validar_secciones_para_hidrosed,
    convertir_secciones_a_hidrosed,
    convertir_puntos_a_hidrosed,
)
from modules_hidrosed.modelacion_avanzada import (
    AdvancedModelParams,
    compute_irregular_hydraulics,
    sensitivity_manning_irregular,
    build_audit_summary,
    water_density_kgm3,
    roughness_estimators,
)


def main():
    data = generar_datos_demo_v13(n_secciones=6, puntos_por_seccion=13)
    errores, advertencias, df_validado = validar_secciones_para_hidrosed(data['df_secciones_validas'], data['df_puntos_secciones'])
    assert not errores, errores
    h03 = convertir_secciones_a_hidrosed(df_validado, caudal_default=80.0, manning_default=0.035)
    h04 = convertir_puntos_a_hidrosed(data['df_puntos_secciones'], df_validado)
    p = AdvancedModelParams(rho_w=water_density_kgm3(20), rho_s=2650, theta_c=0.047, temp_c=20)
    res = compute_irregular_hydraulics(h03, h04, params=p, return_period_years=100)
    assert len(res) == len(h03)
    required = ['Cota_lamina_agua_m', 'y_critico_m', 'Fr', 'Gs_MPM_m3s', 'Gs_EngelundHansen_m3s', 'Socavacion_ajustada_m', 'Estado_QA']
    for c in required:
        assert c in res.columns, c
    sens = sensitivity_manning_irregular(h03, h04, params=p)
    audit = build_audit_summary(res, sens)
    assert not audit.empty
    nvals = roughness_estimators(32, 64, 90)
    assert nvals['n_recomendado_grano_mediana'] > 0
    print('OK modelacion v5', len(res), 'secciones', 'audit rows', len(audit))

if __name__ == '__main__':
    main()
