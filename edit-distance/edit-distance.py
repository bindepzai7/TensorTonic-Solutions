def edit_distance(s1, s2):
    """
    Compute the minimum edit distance between two strings.
    """
    # s1, s2 = s1.split(), s2.split()
    n1, n2 = len(s1), len(s2)
    dp = [[0]*(n2+1) for _ in range(n1+1)]
    for i in range(0, n1+1):
        dp[i][0] = i
    for i in range(0, n2+1):
        dp[0][i] = i
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[n1][n2]