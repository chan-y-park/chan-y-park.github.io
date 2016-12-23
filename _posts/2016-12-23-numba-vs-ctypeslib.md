---
layout: post
title: Comparison between Numba and ``numpy.ctypeslib``
---

{% highlight python %}
def _f_df_at_zx(N, phi_k_n_czes, phi_k_d_czes, z_0, x_0):
    """
    Calculate f(z_0, x_0) and df(z_0, x_0)/dx,
    which are used in Newton's method
    to find the root of f(z_0, x) near x = x_0.

    phi_k_czes = (N, phi_k_n_czes, phi_k_d_czes),
    phi_k_n_czes = [(k, c_n, e_n), ...], 
    phi_k_d_czes = [(k, c_d, e_d), ...], 
    where
        f(z, x) = x^N + ... + (c_n*z^e_n + ...) / (c_d*z^e_d + ...) * x^k + ...
    """
    f_0 = x_0 ** N
    df_0 = N * (x_0 ** (N - 1))
    phi_ns = numpy.zeros(N, dtype=numpy.complex128)
    phi_ds = numpy.zeros(N, dtype=numpy.complex128)

    for k, c, e in phi_k_n_czes:
        phi_ns[k] += c * (z_0 ** e)

    for k, c, e in phi_k_d_czes:
        phi_ds[k] += c * (z_0 ** e)

    for k in range(N):
        phi_k = phi_ns[k] / phi_ds[k]
        f_0 += phi_k * (x_0 ** k)
        if k > 0:
            df_0 += k * phi_k * x_0 ** (k - 1)

    return f_0, df_0
{% endhighlight %}

{% highlight c %}
newton_params f_df_dx_0(
    int N,
    diff_params phi_n,
    diff_params phi_d,
    double complex z_0,
    double complex x_0
) {
    double complex phi_[N][2];
    double complex f_0 = cpow(x_0, N);
    double complex df_dx_0 = N * cpow(x_0, N - 1);

    int i;
    int k;
    double complex c;
    double e;

    double complex phi_k;

    newton_params newton;

    for(i = 0; i < N; i++) {
        phi_[i][0] = 0;
        phi_[i][1] = 0;
    }

    for(i = 0; i < phi_n.n; i++) {
        k = phi_n.k[i];
        c = phi_n.c[i];
        e = phi_n.e[i];
        phi_[k][0] += c * cpow(z_0, e);
    }

    for(i = 0; i < phi_d.n; i++) {
        k = phi_d.k[i];
        c = phi_d.c[i];
        e = phi_d.e[i];
        phi_[k][1] += c * cpow(z_0, e);
    }

    for(k = 0; k < N; k++) {
        phi_k = phi_[k][0] / phi_[k][1];
        f_0 += phi_k * cpow(x_0, k);
        if (k > 0) {
            df_dx_0 += k * phi_k * cpow(x_0, (k - 1));
        }
    }
    newton.f = f_0;
    newton.df_dx = df_dx_0;
    return newton;
}
{% endhighlight %}