### Computer vision with Pytorch


[![Typing SVG](https://readme-typing-svg.herokuapp.com?color=231AF7&lines=My+CV+code)](https://git.io/typing-svg)

def Adam():
    t = 0; mt = 0; vt = 0
    while t < Tmax or not converged:
        t += 1
        gt = gradient(gt)
        mt = beta1 * mt + (1 - beta1) * gt
        vt = beta2 * vt + (1 - beta2) * gt**2
        Bias correction
        # mt = mt / (1 - beta1 ** t)
        # vt = vt / (1 - beta2 ** t)
        Update
        wt = wt - alpha * mt / (sqrt(vt) + epsilon) 