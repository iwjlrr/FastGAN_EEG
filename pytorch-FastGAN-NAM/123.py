ndf=64
nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
nfc = {}
for k, v in nfc_multi.items():

    nfc[k] = int(v * ndf)
    print(nfc)
    print(nfc[k])
