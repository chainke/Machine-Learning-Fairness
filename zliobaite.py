import numpy
import math
import scipy.stats as stats

param_cik = 100
param_n = 10000
# param_pF = 0.2
# param_pp = 0.7
file_name_start = 'results/out_'
param_table_cap = 9



def generate_data(n, pF, pp, disc):
    """
        Generates data array according to paper.
        This happens by generating two sets:
        Set 1 makes up ?% and is created completely in a random and not discrimination way.
        Set 2 makes up 100 - ?% and is created in a maximal discriminating way.

        Parameters
        ----------
        n: int
            number of individuals
        pF: float
            proportion of females
        pp: float
            proportion of positive outcome
        disc: float
            the underlying discrimination
            eg. 0.5 for 50% discrimination rate

        Returns
        -------
        data:
            data set with given discrimination rate
            three rows containing:
    """

    # number of females
    nF = round(pF * n)
    # number of males
    nM = n - nF

    # number of positive outcomes
    npos = round(pp * n)

    if (disc < 0):
        do_reverse = 1
        disc = -disc
    else:
        do_reverse = 0


    # data = runif(n, 0, 1)
    data = list(numpy.random.uniform(0, 1, n))

    # s is the list of the combination of a list with nF times 'F' and a list with NM times 'M'
    # s = c(rep('F', nF), rep('M', nM))
    s = 'F' * nF + 'M' * nM

    nswapF = round(nF * disc)
    nswapM = round(nM * disc)

    if (nswapF + nswapM) > 0:
        # ind_pick is a list of chosen indices by
        # ind_pick = c(1: nswapF, (nF + 1): (nF + nswapM))
        # ind_pick = 1: nswapF +  (nF + 1): (nF + nswapM)
        ind_pick = range(1, nswapF + 1) + range(nF + 1, nF + nswapM + 1)

        # data_pick = data[ind_pick]
        # choose elements of chosen indices from data
        data_pick = [data[x] for x in ind_pick]

        # list of indices in ascending order from data_pick
        # ind = order(data_pick)
        ind = sorted(data_pick)
        # sorted of list in ascending order
        # ind_pick_sorted = ind_pick[ind]
        ind_pick_sorted = [ind_pick[x] for x in ind]

        # TODO re-compute all variables
        if (do_reverse):
            s[ind_pick_sorted] = c(rep('M', nswapM), rep('F', nswapF))
        else:
            s[ind_pick_sorted] = c(rep('F', nswapF), rep('M', nswapM))



    ind = order(data, decreasing=TRUE)
    data = data[ind]
    s = s[ind]

    c = rep(0, n)
    c[1:npos] = 1

    data = cbind(data, c, s)
    colnames(data) = c('y', 'c', 's')
    return (data)


# Use scipy.stats.entropy instead!
def compute_ent (xx)  # already a table
    # ind = which(xx > 0) # Look which indices are greater than 0
    # xx = xx[ind] # Take those
    # H = -sum(math.log2(xx) * xx) # Calculate what?

    xx_pos = [x for x in xx if x > 0]
    H = -sum([math.log2(x) * x for x in xx_pos])
    return (H)

# TODO:
def measure_disc(data):

    n = dim(data)[1]
    ps = table(data[, 's']) / n
    pc = table(data[, 'c']) / n
    pjoint = table(data[, 'c'], data[, 's']) / n

    ddif = pjoint['1', 'M'] / ps['M'] - pjoint['1', 'F'] / ps['F']

    if (ddif > 0):

        m1 = pc['1'] / ps['M']
        m2 = pc['0'] / ps['F']
    else:
        m1 = pc['1'] / ps['F']
        m2 = pc['0'] / ps['M']

    dmax = min(m1, m2)
    ddnorm = ddif / dmax

    dratio = (pjoint['1', 'M'] / ps['M']) / (pjoint['1', 'F'] / ps['F'])
    delift = (pjoint['1', 'M'] / ps['M']) / pc['1']
    dolift = (pjoint['1', 'M'] / pjoint['1', 'F']) / (pjoint['0', 'M'] / pjoint['0', 'F'])

    Hc = stats.entropy(pc)
    Hs = stats.entropy(ps)
    MI = Hc + Hs - stats.entropy(pjoint)
    MInorm = MI / math.sqrt(Hc * Hs)

    dd = cbind(ddif, ddnorm, dratio, delift, dolift, MInorm)
    return (dd)

for pF in [x / 10.0 for x in range(1, 11, 2)]:
# for (pF in seq(0.1, 0.9, 0.2)):

    for pp in [x / 10.0 for x in range(1, 11, 2)]:
    # for (pp in seq(0.1, 0.9, 0.2)):

            # Initialise array?
            # ddall = c()

            for disc in [x / 10.0 for x in range(-10, 11, 1)]:
            # for (disc in seq(-1, 1, 0.1)):

                # Initialise?
                # dd = c()

                # TODO:
                for sk2 in range(1, param_cik + 1):
                # for (sk2 in 1: param_cik):

                    data1 = generate_data(param_n, pF, pp, disc)
                    dd = rbind(dd, measure_disc(data1))

                dd = round(apply(dd, 2, mean), digits=3)
                ddall = rbind(ddall, c(disc, dd))

        table_cap_neg = -param_table_cap
        colnames(ddall)[1] = 'disc'
        ind = which(ddall == Inf)
        ddall[ind] = param_table_cap
        ind = which(ddall == -Inf)
        ddall[ind] = table_cap_neg
        ind = which(ddall > param_table_cap)
        ddall[ind] = param_table_cap
        ind = which(ddall < table_cap_neg)
        ddall[ind] = table_cap_neg
        file_name_now = paste(file_name_start, 'pF',as.character(pF * 100), '_pp',as.character(
            pp * 100), '.dat', sep = '')
        print(file_name_now)
        write.table(ddall, file=file_name_now, row.names = FALSE, col.names = TRUE, sep = ' ', quote = FALSE)

