import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Параметры стандартной атмосферы
def get_atm_params(H):
    '''
    Расчет параметров стандартной атмосферы
        Аргументы:
            - H (float) - геометрическая высота [0 м...13000 м]
        
        Результат:
            - parametrs (dict):
                {'T': (float) - статическая температура, К,
                 'p': (float) - статическое давление, Па,
                 'rho': (float) - плотность воздуха, кг/м^3,
                 'a': (float) - скорость звука в воздухе, м/с}
    '''

    atm_param_data = np.array([[0, 288.150, 101325, 1.22500, 340.294],
    [250, 286.525, 98357.6, 1.19587, 339.333],
    [500, 284.900, 95461.3, 1.16727, 338.370],
    [750, 283.276, 92634.6, 1.13921, 337.403],
    [1000, 281.651, 89876.3, 1.11166, 336.435],
    [1500, 278.402, 84559.7, 1.05810, 334.489],
    [2000, 275.154, 79501.4, 1.00655, 332.532],
    [2500, 271.906, 74692.7, 0.956954, 330.563],
    [3000, 268.659, 70121.2, 0.909254, 328.584],
    [3500, 265.413, 65780.4, 0.863402, 326.592],
    [4000, 262.166, 61660.4, 0.819347, 324.589],
    [4500, 258.921, 57752.6, 0.777038, 322.573],
    [5000, 255.676, 54048.3, 0.736429, 320.545],
    [5500, 252.431, 50539.8, 0.697469, 318.505],
    [6000, 249.187, 47217.6, 0.660111, 316.452],
    [6500, 245.943, 44075.5, 0.624310, 314.385],
    [7000, 242.700, 41105.3, 0.590018, 312.306],
    [7500, 239.457, 38299.7, 0.557192, 310.212],
    [8000, 236.215, 35651.6, 0.525786, 308.105],
    [8500, 232.974, 33154.2, 0.495757, 305.984],
    [9000, 229.733, 30800.7, 0.467063, 303.848],
    [9500, 226.492, 28584.7, 0.439661, 301.697],
    [10000, 223.252, 26499.9, 0.413510, 299.532],
    [10500, 220.013, 24540.2, 0.388570, 297.351],
    [11000, 216.650, 22699.9, 0.364801, 295.154],
    [11500, 216.650, 20984.7, 0.337439, 295.069],
    [12000, 216.650, 19399.4, 0.311937, 295.069],
    [12500, 216.650, 17934.0, 0.288375, 295.069],
    [13000, 216.650, 16579.6, 0.266595, 295.069]])
    if H<0:
        if H<-1000:
            H = atm_param_data[:,0][0] 
        else:
            H = atm_param_data[:,0][0]    
    if H > atm_param_data[:,0][-1]:
        H = atm_param_data[:,0][-1]
    T_atm = interp1d(atm_param_data[:,0], atm_param_data[:,1])
    p_atm = interp1d(atm_param_data[:,0], atm_param_data[:,2])
    rho_atm = interp1d(atm_param_data[:,0], atm_param_data[:,3])
    a_atm = interp1d(atm_param_data[:,0], atm_param_data[:,4])
    parametrs = {'T': T_atm(H),
                 'p': p_atm(H),
                 'rho': rho_atm(H),
                 'a': a_atm(H)}
    return parametrs

# Зависимость приведенной скорости потока от числа Маха
def get_lam_M(k, M):
    """
    Расчет приведенной скорости полета
        Аргументы:
            - M (float) - число Маха;
            - k (float) - показатель адиабаты;
        Результат:
            - lam (float) - приведенная скорость полета.
    """
    return np.sqrt((((k+1) / 2) * M**2) / (1+((k-1) / 2)*M**2))

# Газодинамическая фунция pi(lambda)
def get_pi(k, lam):
    """
    Газодинамическая функция pi
        Аргументы:
            - k (float) - показатель адиабаты;
            - lam (float) - приведенная скорость полета;
        Результат:
            - pi (float) - значение ГД функции.
    """
    return (1-((k-1) / (k+1))*lam**2) ** (k/(k-1))

# газодинамическая функция q(lambda)
def get_q(k, lam):
    """
    Газодинамическая функция q
        Аргументы:
            - k (float) - показатель адиабаты;
            - lam (float) - приведенная скорость полета;
        Результат:
            - q (float) - значение ГД функции.
    """
    return (((k+1)/2)**(1/(k-1)))*((1-((k-1)*lam**2)/(k+1))**(1/(k-1)))*lam

def get_tau(k, lam):
    """
    Газодинамическая функция tau
        Аргументы:
            - k (float) - показатель адиабаты;
            - lam (float) - приведенная скорость полета;
        Результат:
            - tau (float) - значение ГД функции.
    """
    return 1 - ((k-1) * lam**2) / (k+1)

# Термодмнамические параметры воздуха и газа
# тдх воздуха 
k = 1.4 # Показатель адиабаты воздуха
R = 287 # Газовая постоянная воздуха
cp = 1005 # Теплоемкость воздуха

# тдх газа
k_g = 1.33 # Показатель адиабаты воздуха
R_g = 293 # Газовая постоянная воздуха
cp_g = 1165 # Теплоемкость воздуха

def get_params_TRDD(init):
    """Расчет термогазодинамических характеристик ТРДД

    Parameters
    ----------
    dict
        словарь следующего вида:
        {
            'M_n': float
                Число Маха полета
            'H': float
                Высота полета, м
            'P': float
                Тяга, Н
            'sigma_vh': float
                К-т восстановления полного давления во вх. устройстве
            'pi_v_zv': float
                Степень повышения давления в вентиляторе
            'eta_v': float
                КПД вентилятора
            'pi_k_zv': float
                Степень повышения полного давления воздуха в компрессоре
            'eta_k': float
                КПД компрессора
            'sigma_kc': float
                К-т восстановления полного давления в основной камере сгорания
            'T_g_zv': float
                Температура торможения за камерой сгорания, К
            'eta_g': float
                К-т полноты сгорания в основной камере
            'eta_tk_zv': float
                КПД турбины компрессора
            'delta_otb': float
                Доля воздуха, отбираемого на охлаждение турбины
            'eta_mk': float
                Механические КПД турбины компрессора
            'eta_tv_zv': float
                КПД турбины вентилятора
            'eta_mv': float
                Механический КПД турбины вентилятора
            'm': float
                Степень двухконтурности
            'sigma_II': float
                К-т восстановления полного давления в наружном контуре
            'phi_cI': float
                К-т скорости при истечении из сопла внутреннего контура
            'phi_cII': float
                К-т скорости при истечении из сопла наружного контура
            'H_u': float
                Удельная теплота сгорания керосина, кДж/кг  
        }

    Returns
    -------
    dict
        словарь следующего вида:
        {
            'P_ud': float
                Удельная тяга двигателя, м/с
            'C_ud': float
                Удельный расход топлива, ?
            'G_v': float
                Расход воздуха через двигатель, кг/с
        }
    """

    # Распаковка словаря
    M_n, H, P, sigma_vh, pi_v_zv, eta_v, pi_k_zv, eta_k, sigma_kc, T_g_zv, eta_g, eta_tk_zv, delta_otb, eta_mk, eta_tv_zv, eta_mv, m, sigma_II, phi_cI, phi_cII, H_u, YEAR = \
    list((init.values()))
    
    a = get_atm_params(H)['a'] # Скорость звука на высоте H
    p_n = get_atm_params(H)['p'] # Статическое давление на высоте H
    T_n = get_atm_params(H)['T'] # Статическая температура на высоте Н
    V_n = M_n * a # Число Маха
    lambda_n = get_lam_M(k, M_n)
    p_v_zv = p_n*sigma_vh/get_pi(k, lambda_n)
    T_v_zv = T_n / get_tau(k, lambda_n)
    p_vn_zv = p_v_zv * pi_v_zv
    T_vn_zv = T_v_zv * (1 + (pi_v_zv**((k-1) / k) - 1) / eta_v)
    p_k_zv = p_vn_zv * pi_k_zv
    T_k_zv = T_vn_zv * (1 + (pi_k_zv**((k-1) / k) - 1) / eta_k)
    p_g_zv = p_k_zv * sigma_kc
    def get_q_t(T_g_zv, T_k_zv, H_u=42900, T0=293):
        '''
        Относительный расход топлива в камере сгорания (формула Я.Т. Ильичева)
            Аргументы:
                T_g_zv (float) - температура газа перед турбиной, К
                T_k_zv (float) - температура торможения за компрессором, К
            Результат:
                q_t (float) - относительный расход в камере сгорания
        '''
        def get_cpT_zv(T_zv):
            return 4*1e-5*T_zv**2+1.0795*T_zv-59.843
        def get_cp_p_zv(T_zv):
            return 0.0004*T_zv**2+2.1436*T_zv-281.82
        q_t=(get_cpT_zv(T_g_zv) - get_cpT_zv(T_k_zv))/(H_u*eta_g-get_cp_p_zv(T_g_zv)+get_cp_p_zv(T0))
        return q_t
    q_t = get_q_t(T_g_zv, T_k_zv)
    X_tk = (cp* (T_k_zv - T_v_zv)) / (cp_g*T_g_zv*eta_tk_zv*(1+q_t)*(1-delta_otb)*eta_mk)
    pi_tk_zv = (1 - X_tk)**(-k_g / (k_g-1))
    p_tk_zv = p_g_zv / pi_tk_zv
    T_tk_zv = T_g_zv * (1 - X_tk*eta_tk_zv)
    X_tv = (cp* (1 + m) * (T_vn_zv - T_v_zv)) / (cp_g*T_tk_zv*eta_tv_zv*(1 + q_t)*(1 - delta_otb)*eta_mv)
    pi_tv_zv = (1 - X_tv)**(-k_g / (k_g-1))
    p_t_zv = p_tk_zv / pi_tv_zv
    T_t_zv = T_tk_zv * (1 - X_tv*eta_tv_zv)
    def for_lam_cIs(lam):
        return - p_n/p_t_zv + get_pi(k_g, lam)
    lam_cIs = fsolve(for_lam_cIs, 0.5)[0]
    if (lam_cIs <= 1):
        type_rash = 'полное расширение'
        lam_cI = lam_cIs * phi_cI
        p_cI_zv = p_n / get_pi(k_g, lam_cI)
        p_cI = p_n
    elif (lam_cIs > 1):
        type_rash = 'неполное расширение'
        lam_cIs = 1
        p_cI = p_t_zv * get_pi(k_g, 1)
        lam_cI = phi_cI
        delta_p_cI = p_cI - p_n
        p_cI_zv = p_cI / get_pi(k_g, lam_cI)
    c_cI = np.sqrt(2*k_g*R_g/(k_g+1)) * lam_cI * np.sqrt(T_t_zv)
    c_cI_ = c_cI + ((np.sqrt(T_t_zv) * (p_cI - p_n)) / (0.0397 * get_q(k_g, lam_cI) * p_cI_zv))
    P_ud_I = (1 + q_t*(1 - delta_otb))*c_cI_ - V_n
    p_II_zv = p_vn_zv * sigma_II
    def for_lam_cII_s(lam):
        return get_pi(k_g, lam) - p_n/p_II_zv
    lam_cII_s = fsolve(for_lam_cII_s, 1)[0]
    if (lam_cII_s <= 1):
        type_rash_II = 'полное расширение в сопле'
        lam_cII = lam_cII_s * phi_cII
        p_cII_zv = p_n / get_pi(k_g, lam_cII)
        p_cII = p_n
    elif (lam_cII_s > 1):
        type_rash_II = 'НЕполное расширение в сопле'
        lam_cII_s = 1
        p_cII = p_II_zv * get_pi(k_g, 1)
        lam_cII = phi_cII
        delta_p_cII = p_cII - p_n
        p_cII_zv = p_cII / get_pi(k_g, lam_cII)
    c_cII = np.sqrt(2*k_g*R_g / (k_g+1)) * lam_cII * np.sqrt(T_t_zv)
    c_cII_ = c_cII + ((np.sqrt(T_vn_zv) * (p_cII - p_n)) / (0.0404 * get_q(k_g, lam_cII) * p_cII_zv))
    P_ud_II = c_cII_ - V_n
    P_ud = (P_ud_I + m*P_ud_II) / (1 + m)
    C_ud = 3600 * (q_t * (1 - delta_otb)) / (P_ud * (1 + m))
    G_v = P / P_ud
    L_e = 0.5 * ((1 + q_t * (1-delta_otb))*c_cI_**2 + m*c_cII_**2 - (1+m)*V_n**2)
    eta_e = L_e / (q_t * (1 - delta_otb) * H_u * 1e3)
    eta_n = ((1 + m) * P_ud * V_n) / L_e
    eta_o = eta_e * eta_n
    result={}
    result['P_ud'] = P_ud
    result['C_ud'] = C_ud
    result['G_v'] = G_v
    result['T_array'] = [T_v_zv, T_vn_zv, T_k_zv, T_g_zv, T_tk_zv, T_k_zv]
    result['T_sections'] = ['в', 'вн', 'к', 'г', 'тк', 'т']
    result['p_array'] = [p_v_zv, p_vn_zv, p_k_zv, p_g_zv, p_tk_zv, p_t_zv, p_cI_zv, p_cII_zv]
    result['p_sections'] = ['в', 'вн', 'к', 'г', 'тк', 'т', 'c1', 'c2']
    return result