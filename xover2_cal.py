import numpy as np
import time as tm
import matplotlib as plt
import os
import netCDF4 as nc
from scipy import interpolate
import shutil
# ------------线性插值有问题使用三次样条插值-------------------

# --------------拟合的南北区域范围的一半----------------------
def dlat_half(lat0):
    P=np.matrix([1.747e-11,3.659e-12,-2.905e-07,-3.81e-08,0.002313,9.786e-05,-8.427])
    P0=P.T
    # -----------------拟合参数-------------------------------------------
    bb=np.matrix([0,0,0,0,0,0,0])
    # ----------------系数-----------------------------
    for i in range(0,7):
        bb[0,i]=lat0**(6-i)
    dlat_h=abs(bb*P0/2.0)
    return dlat_h
# ------------read nc-------------------------------
def read_karin(nc_file):
    # ------------time用编号代替--------------
    nc_data = nc.Dataset(nc_file)
    # 将nc文件的数据传入data
    # time = nc_data.variables["time"][:].data
    lat = nc_data.variables["latitude"][:][:].data
    lon = nc_data.variables["longitude"][:][:].data
    js = nc_data.variables["alongjs"][:][:].data
    # ssh = nc_data.variables["ssh"][:][:].data
    return js, lon, lat


# ------------------------------------用于计算交叉点子程序1----------------------------------------------------------
# ----------------寻找两个数组X1和X2的公共区域--------
def numpy_find_commen_domain(X1, X2):
    # X1X2变为单调递增
    # X1.sort()
    # X2.sort()
    # 由于本研究X1,X2为单调的因此进行了改进：
    if X1[1] < X1[0]: X1 = np.flipud(X1)
    if X2[1] < X2[0]: X2 = np.flipud(X2)
    a = np.max([X1[0], X2[0]])
    b = np.min([X1[-1], X2[-1]])

    X_full = np.append(X1, X2)
    X_full = np.array(list(set(X_full)))  # 去重
    X_full.sort()
    X_commen = X_full[np.where(np.logical_and(X_full >= a, X_full <= b))]
    return X_commen


# ------------------------------------用于计算交叉点子程序2----------------------------------------------------------
# -------------------判断离散曲线与0轴的交点-----------------------
def numpy_scipy_find_roots_by_XY(X, Y):
    # X必须为严格单调递增数据
    if np.all(np.diff(X) > 0):
        pass
    else:
        raise Exception('X必须为严格单调递增数据!')

    # 根据离散点XY构造样条曲线
    # BSpline(builtins.object)
    # tck: A spline, as returned by `splrep` or a BSpline object.
    tck = interpolate.make_interp_spline(x=X, y=Y, k=3)  # 样条插值k=3

    # tuple(object)
    # tck: A spline, as returned by `splrep` or a BSpline object.
    # tck = interpolate.splrep(x=X, y=Y, k=1, s=0) # roots = [ 3.         15.53846154 16.66666667]
    # k为样条曲线的阶数，k=1为线性，k=2为二次多项式，...

    # 转换
    # class PPoly(_PPolyBase); Construct a piecewise polynomial from a spline
    piecewise_polynomial = interpolate.PPoly.from_spline(tck, extrapolate=None)

    # 求根
    roots_X_ = piecewise_polynomial.roots()  # class ndarray(builtins.object)
    # 在X范围内筛选根
    roots_X = roots_X_[np.where(np.logical_and(roots_X_ >= X[0], roots_X_ <= X[-1]))]
    return roots_X


# ------------------------------------两条单升降轨的交叉点----------------------------------------------------------
def single_ad_cross(X1, Y1, X2, Y2):
    # 计算两线段之间的交叉点
    # 输入：列向量：X1,Y1 第一条线段的坐标；X2,Y2第2条线段的坐标
    # 输出：intersections_X, intersections_Y 若无交叉点则是空的
    intersections_X = np.array([])
    intersections_Y = np.array([])
    # -----------------------判断有没有lon交叉，没有则不用求--------------------------------
    Y1max = max(Y1[0], Y1[Y1.size-1])
    Y1min = min(Y1[0], Y1[Y1.size-1])
    Y2max = max(Y2[0], Y2[Y2.size-1])
    Y2min = min(Y2[0], Y2[Y2.size-1])
    if Y1max < Y2min or Y1min > Y2max:
        return intersections_X, intersections_Y

    # ----------------------------------------------------------
    X1max = max(X1[0], X1[X1.size-1])
    X1min = min(X1[0], X1[X1.size-1])
    X2max = max(X2[0], X2[X2.size-1])
    X2min = min(X2[0], X2[X2.size-1])
    if X1max < X2min or X1min > X2max:
        return intersections_X, intersections_Y
    # ----------------------------------------------------------------

    # 要求X1和X2都是单调递增
    if np.all(np.diff(X1) > 0):
        pass
    else:
        X1= np.flipud(X1)
        Y1=np.flipud(Y1)

    if np.all(np.diff(X2) > 0):
        pass
    else:
        X2= np.flipud(X2)
        Y2= np.flipud(Y2)

    X_all = numpy_find_commen_domain(X1, X2) # X_all为单调增
    if X_all.size > 3:                 # 后面呢采用3次样条插值，所以最小为4
        # -----得到两条线X坐标重合的区域X_all------------------
        # -----原有线性插值有问题改成3次样条----------------------
        # Y1_new2 = np.interp(X_all, X1, Y1)
        # Y2_new2 = np.interp(X_all, X2, Y2)
        # 进行三次样条拟合
        ipo3_1 = interpolate.splrep(X1, Y1)  # 样本点导入，生成参数
        Y1_new2 = interpolate.splev(X_all, ipo3_1)  # 根据观测点和样条参数，生成插值
        ipo3_2 = interpolate.splrep(X2, Y2)  # 样本点导入，生成参数
        Y2_new2 = interpolate.splev(X_all, ipo3_2)  # 根据观测点和样条参数，生成插值
        # --------------------------------------------------------
        # -----分别依据两条线差值出 X_all对应的Y,当二者相等的时候我们就认为是交叉点的位置------
        deta_Y = Y1_new2 - Y2_new2
        # -------计算(X_all,detaY)与0线之间的交点就是交叉点--------------------------
        intersections_X = numpy_scipy_find_roots_by_XY(X_all, deta_Y)

        if intersections_X.size>0:
            intersections_Y = interpolate.splev(intersections_X, ipo3_1)
        if intersections_X.size>1:
            print('warning: two intersections!!!!!!!!!!!!!')
            os.system("pause")
    return intersections_X, intersections_Y


# ------------宽刈幅升降轨交点位置:并记录在文件中----------------------------------
def swath_ad_cross(time1, lon1, lat1, time2, lon2, lat2, fta, file2):

    lonmin = 105.0
    lonmax = 125.0
    latmin = 0.0
    latmax = 30.0

    for i in range(lat1[:, 0].size):
        # if lon1[i
        for j in range(lat2[:, 0].size-1,-1,-1):
            # ------------由于crosstrack lon并不是单调增需要调整----------------

            intersections_lat,  intersections_lon = single_ad_cross(lat1[i, :], lon1[i, :],lat2[j, :], lon2[j, :])
            if intersections_lat.size > 0:
                mnum = 0
                for m in range(lat1[0, :].size - 1):
                    if (lat1[i, m] - intersections_lat[0]) * (lat1[i, m + 1] - intersections_lat[0]) <= 0:
                        mnum = m
                        break
                nnum = 0
                for n in range(lat2[0, :].size - 1):
                    if (lat2[j, n] - intersections_lat[0]) * (lat2[j, n + 1] - intersections_lat[0]) <= 0:
                        nnum = n
                        break
                # -------------只保留范围内的交叉点--------------------------------------
                if lonmin<intersections_lon<lonmax and latmin<intersections_lat<latmax:
                    print('%d  %f  %f  %d  %s  %d  %d' %(time1[i], intersections_lon, intersections_lat, mnum, file2[:-3], time2[j], nnum),file=fta)
                #fta.close()
                # print(mnum)

    return


# ---------------------确定在交叉点所在的范围内的数据-------------------
def karin_local(time0, lon, lat, lonmin, lonmax, latmin, latmax):
    j_flag = np.ones(lon[:, 0].size)
    # 符合条件的个数
    nums = 0  # 记录有值的
    index_list = []
    for i in range(0, lon[:, 0].size):
        for j in range(0, lon[0, :].size):
            if lonmin < lon[i, j] < lonmax or latmin < lat[i, j] < latmax:
                j_flag[i] = 0
                break
        if j_flag[i] == 0:
            nums = nums + 1  # 有数据在此范围内
            index_list.append(i)  # 记录有数的点
    if nums > 0:
        time_f = np.empty(nums)
        lon_f = np.empty((nums, lon[0, :].size))
        lat_f = np.empty((nums, lon[0, :].size))
        im = -1
        for ii in index_list:
            im = im + 1
            for j in range(0, lon[0, :].size):
                time_f[im] = time0[ii]
                lon_f[im, j] = lon[ii, j]
                lat_f[im, j] = lat[ii, j]
    return time_f, lon_f, lat_f


# -------------判断宽刈幅pass间是否有交叉点，返回交叉点坐标lon,lat----------------------------------------------
def pd_cross(lon1, lat1, lon2, lat2):
    # 输入二组矩阵 宽刈幅pass A lon1 lat1 宽刈幅pass B lon2 lat2
    # 输出 交叉点所在范围的lonmin和lonmax或者latmin和latmax 若无交叉点四者为nan

    lonmin = np.nan
    lonmax = np.nan
    latmin = np.nan
    latmax = np.nan

    jcd_lat, jcd_lon = single_ad_cross(lat1[:, 0], lon1[:, 0], lat2[:, 68], lon2[:, 68])
    if jcd_lon.size > 0:
        latmin = jcd_lat[0] - dlat_half(jcd_lat[0])
        latmax = jcd_lat[0] + dlat_half(jcd_lat[0])
    else:
        jcd_lat, jcd_lon = single_ad_cross(lat1[:, 68], lon1[:, 68], lat2[:, 0], lon2[:, 0])
        if jcd_lon.size > 0:
            latmin = jcd_lat[0] - dlat_half(jcd_lat[0])
            latmax = jcd_lat[0] + dlat_half(jcd_lat[0])
    return lonmin, lonmax, latmin, latmax  # 最后return 有可能包括4个nan，表明无交叉点 lat有值表示有


if __name__ == "__main__":
    # ----------------------原始文件lat扩大了，为了保证包含菱形区域的一个角点--------------------
    t1 = tm.time()
    a_filepath = 'G:\\pchdata\\region1\\ncfile\\cycle_00n'
    d_filepath = 'G:\\pchdata\\region1\\ncfile\\cycle_00n'
    cross_save_path_z = 'G:\\pchdata\\region1\\crosslocal'
    if os.path.exists(cross_save_path_z):
        shutil.rmtree(cross_save_path_z)
    os.makedirs(cross_save_path_z)
    a_files = os.listdir(a_filepath)
    d_files = os.listdir(d_filepath)
    for a_file in a_files:
        print(a_file)
        cnum = 0  # 该pass与其他pass的总共的交叉点数，用于控制文件的打开
        anum = int(a_file[4:7])
        d_files.remove(a_file)
        # if (a_file!='pass118.nc'): continue
        a_file_f = a_filepath + '\\' + a_file
        js1, lon1, lat1 = read_karin(a_file_f)  # 读取apass文件
        cross_save_file_z_a = cross_save_path_z + '\\' + a_file[:-3] + '.txt'  # 储存a交叉点文件
          # 保证不重复
        for d_file in d_files:
            print(d_file)
            d_file_f = d_filepath + '\\' + d_file
            js2, lon2, lat2 = read_karin(d_file_f)  # 读取dpass文件
            dnum = int(d_file[4:7])

            if (abs(anum - dnum) % 2) == 0:
                continue  # 同为升轨或者降轨
            else:
                lonmin, lonmax, latmin, latmax = pd_cross(lon1, lat1, lon2, lat2)
                if np.isnan(latmin):
                    continue  # 无交点
                else:
                    js1_f, lon1_f, lat1_f = karin_local(js1, lon1, lat1, lonmin, lonmax, latmin, latmax)
                    js2_f, lon2_f, lat2_f = karin_local(js2, lon2, lat2, lonmin, lonmax, latmin, latmax)
                    cnum = cnum + 1
                    if cnum == 1: fta = open(cross_save_file_z_a, 'a')
                    swath_ad_cross(js1_f,lon1_f, lat1_f, js2_f,lon2_f,  lat2_f,   fta, d_file)
        if cnum > 0: fta.close()
    t2=tm.time()
    print(t2-t1)
