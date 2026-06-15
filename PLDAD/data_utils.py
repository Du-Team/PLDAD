import numpy as np

from sklearn.preprocessing import MinMaxScaler

def build_sliding_windows(data, seq_length, step=1):
    if data.shape[0] < seq_length:
        return np.empty((0, seq_length, data.shape[1]), dtype=data.dtype)

    windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=seq_length, axis=0)

    windows = np.moveaxis(windows, -1, 1)
    if step == 1:
        return np.ascontiguousarray(windows)

    num_samples = (data.shape[0] - seq_length) // step
    if num_samples <= 0:
        return np.empty((0, seq_length, data.shape[1]), dtype=data.dtype)

    indices = np.arange(num_samples) * step
    return np.ascontiguousarray(windows[indices])

def average_filter(values, n=3):
    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res

def spectral_residual_transform(values):  

    EPS = 1e-8

    trans = np.fft.fft(values)  
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)  
    eps_index = np.where(mag <= EPS)[0]  
    mag[eps_index] = EPS

    mag_log = np.log(mag)  
    mag_log[eps_index] = 0

    spectral = np.exp(mag_log - average_filter(mag_log, n=3))

    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    trans.real[eps_index] = 0
    trans.imag[eps_index] = 0

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
    return mag

def proprocess(df):
    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError("Data must be 2-D array")

    if np.any(sum(np.isnan(df)) != 0):
        print("Data contains nan. Will be repalced with 0")

        df = np.nan_to_num()

    df = MinMaxScaler().fit_transform(df)

    print("Data is normalized [0,1]")

    return df

def read_train_data(seq_length, file='', step=1, valid_portition=0.3):
    values = []

    df = np.load('./datasets/train/' + file, allow_pickle=True)
    print(df.shape)

    (whole_len, whole_dim) = df.shape

    for i in range(whole_dim):
        df[:, i] = spectral_residual_transform(df[:, i])  

    print('SR')

    values = proprocess(df)  

    n = int(len(values) * valid_portition)

    if n > seq_length:  
        train, val = values[:-n], values[-n:]
        train_data = build_sliding_windows(train, seq_length, step=step)
        val_data = build_sliding_windows(val, seq_length, step=step)

    else:
        train = values
        train_data = build_sliding_windows(train, seq_length, step=step)
        val_data = train_data

    return train_data, val_data

def read_test_data(seq_length, file=''):
    df = np.load('./datasets/test/' + file, allow_pickle=True)
    label = np.load('./datasets/test_label/' + file, allow_pickle=True).astype(np.float64)
    print(df.shape, label.shape)

    (whole_len, whole_dim) = df.shape

    for i in range(whole_dim):
        df[:, i] = spectral_residual_transform(df[:, i])

    print('SR')

    test = proprocess(df)

    test_data = build_sliding_windows(test, seq_length, step=1)

    return test_data, label
