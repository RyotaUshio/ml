"""Feature Extraction Processors."""

import numpy as np


def pca(X):
    r"""主成分分析を行う.

    Parameters
    ----------
    X : array_like
        データを行として並べた行列

    Returns
    -------
    eigvecs : np.ndarray
        分散共分散行列の固有ベクトルを行として並べた行列. 寄与率の大きい順にソートされている.
    ctrb : np.ndarray
        各固有ベクトルの寄与率を、eigvecsに対応した順番で並べた配列."""
    
    sigma = np.cov(X, rowvar=False) # 分散共分散行列
    eigvals, eigvecs = np.linalg.eig(sigma) # 固有値・固有ベクトル
    # np.linalg.eigは固有ベクトルを列とするarrayを返すので、転置をとる.
    eigvecs = eigvecs.T
    
    # 固有値の大きい順に固有値と固有ベクトルをソートする
    idx = np.flip(np.argsort(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[idx]
    
    ctrb = eigvals / eigvals.sum()  # 各固有ベクトルの寄与率
    
    return eigvecs, ctrb


def project(X, basis):
    r"""与えられたデータ集合を、与えられた基底に射影して変換する。

    Parameters
    ----------
    X : array_like
        データを行として並べた行列
    basis : array_like
        基底を行として並べた行列

    Returns
    -------
    np.ndarray
        射影後のデータを行として並べた行列
    """
    X_projected = np.matmul(X, basis.T)
    if np.all(np.imag(X_projected) == 0):
        return X_projected.astype(float)
    return X_projected
