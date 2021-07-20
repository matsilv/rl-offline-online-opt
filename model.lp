\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0.054 pDiesel_0 - 1000 pStorageIn_0 - 0.04699 pGrid_0 + 0.04699 pGrid_0
Subject To
 Power_balance: pDiesel_0 - pStorageIn_0 + pStorageOut_0 - pGrid_0
   + pGrid_0 = 111.3976700286359
 R1: - pStorageIn_0 + pStorageOut_0 + cap_0 = 800
 R2: cap_0 <= 1000
 R3: pStorageIn_0 <= 200
 R4: pStorageOut_0 <= 800
 R5: pStorageIn_0 <= 200
 R6: pStorageOut_0 <= 200
 R7: pDiesel_0 <= 1200
 R8: pGrid_0 <= 600
Bounds
End
