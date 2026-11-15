# Manifest: haiku-2026-11-15 raw outputs

The run's summary is [docs/research/haiku-2026-11-15/README.md](../../../../docs/research/haiku-2026-11-15/README.md), with its scripts. The case lists in [sets/](sets/) stay here: the compare scripts read them.

The raw files below moved off `main` on 2026-11-15 to keep its tree small; none was lost. They
are on the orphan branch **`data/research-archive`** of this repository, at the same paths
(one commit per run). To get them back next to the summary:

```bash
git fetch origin data/research-archive
git restore --source=origin/data/research-archive -- <path>   # one file, or a folder
```

Check a file against this list with `sha256sum <path>`. 92 files, 952,520 bytes.

| path | bytes | sha256 |
|---|---|---|
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/long57.json` | 6,288 | `e882e2c2c0f86711769ce4341f172242372decda541de5529728c90471a73c9a` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/long57.jsonl` | 203,804 | `8843591c07ae51ac7873a76b562f985c6c1719bba9e27ef01d1352c35bbb6511` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/long57.md` | 1,186 | `d6f84b8c0a7cc3922098a1a9f4f03a4cd772ce7355929c8df73d8b3c12f80f69` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/ordinary.json` | 5,627 | `bfebd63a5c9d77f9052d1863f886c9fc3a528ad7a759e21b0b98cbde7294c13b` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/ordinary.jsonl` | 60,231 | `57487326a386f200e78a7780483803d741eeaf7677cbf9020e74ec56836e8852` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/ordinary.md` | 1,054 | `4bbe4895c65b89d3369118052ed00a32fdaea500149052dc09837a0f9f1c9259` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/sw16.json` | 2,349 | `8e7a46b99b2179490425245439821a12fc9a1c8452d0ed8669cc8b4a3597d2b1` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/sw16.jsonl` | 9,800 | `072baedab296297467251c6b79cca51fe064d268dfa7d2c6cb8f8b0b1bdc638f` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/sw16.md` | 493 | `9f3e64a9ff61f6b1f824622c2d27a9cda6bdc4451e26e9fb240f8f113db5e46e` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/sw8.json` | 2,020 | `1e55b82571f8f0394211d83d94c80ac6653687fa633d004de637ce261135248d` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/sw8.jsonl` | 4,623 | `64e72badd0abeeb4bce62bd9710f6e6b847030d55e6c825bd42b1263cb1df406` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/sw8.md` | 437 | `e99429eacbb12d4d4325e4d6cf70b5cb722c96632890c9b7a833e9b03e59c36a` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/swown.json` | 2,026 | `dfe76b19b83216c5796ed34d0940b259c1d3dfcd700b4f11c4b48328a08140cc` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/swown.jsonl` | 19,966 | `8c1baee4852a8e1110d24cedb82197f3e72744cfed75bcbf5ceffa5fe359f3c7` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/backend-1/swown.md` | 453 | `acd509b5c6dc7d9b253d1db4a9b9e2f2a05c6ab9477ffab1c03d8388bafafa79` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/long57.json` | 6,287 | `622f793e77ca7be356b29833b6e93abc8024131854ab98249dabba1ba9fc3caf` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/long57.jsonl` | 204,629 | `e9e92ceb86980dd1476cba9bb50a40246967c886616d095ab622c2d6b8d06caf` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/long57.md` | 1,186 | `0de4c90459e3be8ba263989914bb29b14d95336f23794d5ab03e518568f142c1` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/ordinary.json` | 5,604 | `8914b82c1c3c47e6bc16561fbe2ba73940052bf2c05c08d1d6fca16b587d2f2a` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/ordinary.jsonl` | 62,964 | `255237bb0baebce98d4fbfd9bb51be977756cd85b5cc82290668ea543846a8db` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/ordinary.md` | 1,054 | `4eca1725499f20f54ef57ec6db8cfe8357c090f0003f031d64753512feffd759` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/sw16.json` | 2,338 | `7d62f22035ea9f55f8acd0ff217ba9918bf4429db209db6a1be04aa98d977063` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/sw16.jsonl` | 10,149 | `256a1669fd375dbbfa504da5d822b22520c0a4be62fc483cab7a9838b432d676` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/sw16.md` | 493 | `c9633771338b6de3946a0b38e61e5a1e650c7051680be13f63a85b2ac3693e9e` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/sw8.json` | 2,011 | `cb235863381a627b9b713a6b51505d744b3226c634eb8b0cc57713c8060ac6a6` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/sw8.jsonl` | 5,769 | `c30bb43ddd8d08ef37332df8fc53aeb5b486e7103707435b31142896465641e1` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/sw8.md` | 437 | `7939b089d1d652d5e4e79f54f06a645ab0a447b45d586f2dc789b686c63d3359` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/swown.json` | 2,026 | `390b070979c0c722567392426a94ecedd89a42f680845333d2890499175c19ff` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/swown.jsonl` | 21,827 | `a19a215882757116aac3aa0718383bfddf70c367e6315232cbb2fc339f96492f` |
| `benchmark/cost/results/haiku-2026-11-15/accuracy/plain-1/swown.md` | 453 | `9a8baa3f20860a04f2329b13d54d08c9e26aa489fcfd2b3da3846b84ae3807e0` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepbare-1/carry.json` | 2,121 | `626da05351b011f5a16ab2b64fdaaeafd234aace5f16070dea6e9d820362fc8a` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepbare-1/carry.jsonl` | 22,192 | `7c298547b20632bf017d8d7ba2dc78c006e0f096eecf32e5d6b69bd3f897b1d4` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepbare-1/carry.md` | 445 | `01941ecb86f173f0fb48d4cbd63f6c68f644885de614c8d9b208d9f755c3eb6f` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepbare-2/carry.json` | 2,107 | `a1c2c90769634b0f89859f226eb605da1e9fe434da5e28783c15c9bf9d71edec` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepbare-2/carry.jsonl` | 23,045 | `b122a9645ded8223e7d83ae577a36ab21a99b7f5912c0d79f4857ed23641b789` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepbare-2/carry.md` | 445 | `b4dcf8415efbc305ac925c588833756e4f3958f8b22a2dd41e64563765643e5a` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepbare-3/carry.json` | 2,134 | `0a99e0df22f8a502082d3062dac967661ecd784b84189725de4f2762c092d768` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepbare-3/carry.jsonl` | 22,456 | `ad48852cd2d2fa41ddfb0ab621f55a7ceeb04f9160ea2854b7cf129641783ff9` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepbare-3/carry.md` | 445 | `fce902ddc62e134c6fd87a530b8c02c0c235247c740771b64d4b44466f3ae021` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepown-1/carry.json` | 2,069 | `2f4fc5ed3a1afe1738a351a83bb0e6c9d936d57de303296ba41ae75b9d6ec49f` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepown-1/carry.jsonl` | 19,440 | `34485816d6c966b6d5fdb85c4bb77b0afca2822deb862bd4a83d3fae58e6f818` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepown-1/carry.md` | 445 | `169cfe61ab0139bb62c4a97eca6170fc771c0ab4e5f4a22e222b93cca96a7bdc` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepown-2/carry.json` | 2,068 | `73bf8d05ce3b1b43746f54647e3b4b3072f21cd705e9667a5b5e91f1ecf0791b` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepown-2/carry.jsonl` | 18,954 | `615f07356cea1f9ddc2daf1da8903c5d81fc9620fd9d4b1a57064f0746dabc11` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepown-2/carry.md` | 445 | `2f8dab2d468415655d318160a5547604ff77edf1396071a4fa58930fd6cf03a9` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepown-3/carry.json` | 2,069 | `9dfdcf4a96c3d118e68f02d89c54306d6ecf4deb1d26606557a622394a223497` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepown-3/carry.jsonl` | 18,564 | `0224f934c093cb889e9aaff141057cd4ab1e5dcfb21316f1153f68fa01218b3b` |
| `benchmark/cost/results/haiku-2026-11-15/carry/deepown-3/carry.md` | 445 | `ec82081c02eb4445f5d3173ee19f8bbff5aa4af3a43ad4b9de29f191cab1fdc5` |
| `benchmark/cost/results/haiku-2026-11-15/carry/reactdeep-1/carry.json` | 2,070 | `b541ec8870ac0981ac3521baf6ab13c64c1114dbb9d206f8f418f9fa781f8fc7` |
| `benchmark/cost/results/haiku-2026-11-15/carry/reactdeep-1/carry.jsonl` | 20,279 | `33c7de52deb023db130edfe8cdab83833c287250dc236c7ec2400031f1fcc166` |
| `benchmark/cost/results/haiku-2026-11-15/carry/reactdeep-1/carry.md` | 440 | `f6dad22c3ee6005bd6e1534cd3f1d3b3ab1946670415d93959b47e05fcde3f39` |
| `benchmark/cost/results/haiku-2026-11-15/carry/reactdeep-2/carry.json` | 2,095 | `6a7a6de286765c1683737cf053ab8680ce01d62a5c9574cf5996a9d24b776433` |
| `benchmark/cost/results/haiku-2026-11-15/carry/reactdeep-2/carry.jsonl` | 21,046 | `0f9b53679fa95fe777f34ba092ce79182c29eee82467a01b6ded2bd6f36dea0e` |
| `benchmark/cost/results/haiku-2026-11-15/carry/reactdeep-2/carry.md` | 440 | `8f71f83359d7ab181b8f4114fef7a9f574943577f779191a380dcdde388f47f6` |
| `benchmark/cost/results/haiku-2026-11-15/carry/reactdeep-3/carry.json` | 2,108 | `e1cf790961ed0965dd46246c03633c1a24fedfd9c31ec94d6fd400620c6608a5` |
| `benchmark/cost/results/haiku-2026-11-15/carry/reactdeep-3/carry.jsonl` | 20,688 | `4f613f049f514566dbef919dc8c2ce8f1962cf533c6af7d0eb93f207f3b5e28e` |
| `benchmark/cost/results/haiku-2026-11-15/carry/reactdeep-3/carry.md` | 440 | `1e3416d1077a3209d8f4f8352bbfd4805f7795e345c02433c33948ce52ef7ebd` |
| `benchmark/cost/results/haiku-2026-11-15/claude/paused-charge/accounted.json` | 824 | `4cda0524b35a563a9048bdacafe26bfdf9a87e8196793844de3adfdb8e8ec665` |
| `benchmark/cost/results/haiku-2026-11-15/claude/paused-charge/approved.json` | 449 | `6f8caee42e9c6ad51d3d0f7794414339173aee8eb72d690295be732e5fc91217` |
| `benchmark/cost/results/haiku-2026-11-15/claude/paused-charge/meter.jsonl` | 3,059 | `7f4aa0d6e8e16fbed309c8fbaca77f7901d019af07d18d1fd3037e7e7162ec26` |
| `benchmark/cost/results/haiku-2026-11-15/claude/paused-charge/rejected.json` | 522 | `d66379b234bcebcdf2e4eca22bd004a1b408409afbc2d63376fa9e958a7e44e0` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run6/accounted.json` | 2,659 | `c70009ec5980ff3695a939382500afabcfa6597edaac34161597db5cd593e2f4` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run6/approved.json` | 448 | `5864d339b3421efaee7a8afb4b42e76dee16ea39c3debae9bfe856e061b4e56e` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run6/background.json` | 2,611 | `079dec8d560233ae574c54d5893643bd8e7791bb6b226722d875d1401a582945` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run6/limit.json` | 744 | `459085dd6e236f2c65063d1bf9c0d4ef19793c299db067e6ed6264f7d9b041c5` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run6/meter.jsonl` | 9,171 | `a918cddf0f04e439c74201d68f6d1b60b06e78e9b2173a6bfc749b3128ae489d` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run6/provider_usage.json` | 336 | `3c6c1d27bc001ed240e045071a4428059940d7f1c86a6e2cc8e2bb4dd8d47e4a` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run6/rejected.json` | 488 | `9870628685fb0cff9fd217555950e4ca8b6a1353966a27e5e968bed4773b3a28` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run6/session.json` | 544 | `78c8f5f7ce1fa122ed7cbf4405252932629e4019d1d12ea1344d498b9d16afeb` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run7/accounted.json` | 1,287 | `c674f3f95167420ac2ab1ef14970ad46f052aa5ab3544cf3da739c34e9dae1be` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run7/background.json` | 2,398 | `0d024400fefa8d2167ded8966c79f3aa6e4a71a112b7f20322a0c54d000aa70e` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run7/meter.jsonl` | 2,624 | `42024c3965656f9a5447b71c7a6e79c7e1daa47eb2e8c08678a795084214c92c` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run7/provider_usage.json` | 332 | `860a4cbcb9fe9ff6b8f7e6cd48dfdb8ee01bcabf6bc413f471beaced0849a9d5` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run8/accounted.json` | 1,278 | `60509870976cac0ac50972d5618834a49fe7a3c65fb3ceab7a76ba460b8f792f` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run8/background.json` | 2,490 | `2ba2c42a1e0aec3e0f284ac5129a31d79a1ade24c3956c95b4c8875ad1be8e05` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run8/meter.jsonl` | 2,624 | `b68b17d900c75b6e39db62260832872a6c6e89f86fcb849fe7034393b195220b` |
| `benchmark/cost/results/haiku-2026-11-15/claude/run8/provider_usage.json` | 332 | `7b7a23d08e016be56d7eb1bdf8db3c2831067961b9b46f0176b41e1172dc73f1` |
| `benchmark/cost/results/haiku-2026-11-15/kimi/cx-kimi.json` | 3,633 | `0470688035ea884663f2ad0f2e646ddfd949f93c74aad81bd8de2937851d30cf` |
| `benchmark/cost/results/haiku-2026-11-15/kimi/cx-kimi.jsonl` | 15,013 | `a01d91ee98877c2b4380ce8476938a7dff84337eaba0922d4cf0aa4f17f78fd4` |
| `benchmark/cost/results/haiku-2026-11-15/kimi/cx-kimi.md` | 695 | `11ac6b2180b922b8110b2904ce8959ba37213c838d884c656f172a1d416f53ea` |
| `benchmark/cost/results/haiku-2026-11-15/langchain-spend/acc-deep.json` | 2,455 | `928e941808c91824336a64f746e7d2d88da472af45a2245d7bd9a335c50cb964` |
| `benchmark/cost/results/haiku-2026-11-15/langchain-spend/acc-main.json` | 2,174 | `e8daa7e0862588a86afbfb03971e2ef7666a1fb373125000a5e863deb0ae1b75` |
| `benchmark/cost/results/haiku-2026-11-15/langchain-spend/cx-deep.json` | 3,319 | `2f5a8629746e5dbbd411b0166d324608490973e83d65298333e78f47534bc1b9` |
| `benchmark/cost/results/haiku-2026-11-15/langchain-spend/cx-deep.jsonl` | 9,751 | `5726a31d3d1b71e0c72342a6b5833a1f2249ecd34d334c9a772fa9203aaf4f67` |
| `benchmark/cost/results/haiku-2026-11-15/langchain-spend/cx-deep.md` | 637 | `533834a745ec27728250b1230a56a7f760444c6bcb389804ec10f5e033fb9c2f` |
| `benchmark/cost/results/haiku-2026-11-15/langchain-spend/cx-main.json` | 3,305 | `310015285e067baae298c897c7da98b31e7c0f4b4d691af65535e39d9311cb83` |
| `benchmark/cost/results/haiku-2026-11-15/langchain-spend/cx-main.jsonl` | 9,472 | `19ed43a3846b01b5b6c012e0a2df395e22d2e9091621899d97410b4076c328e2` |
| `benchmark/cost/results/haiku-2026-11-15/langchain-spend/cx-main.md` | 637 | `b25561a11ab1d1b24bd8afa7bd9b5931973722b414b27368118f54ae597d4329` |
| `benchmark/cost/results/haiku-2026-11-15/langchain-spend/meter-deep.jsonl` | 3,564 | `252bd339a263d82c81fe3973dc187da701575bd3e9be656a45e7d519b8db045f` |
| `benchmark/cost/results/haiku-2026-11-15/langchain-spend/meter-main.jsonl` | 3,149 | `93ef98363549a885cd4d64203035c5e342cb1119b72c6bd54d2b72ee022ce9e4` |
| `benchmark/cost/results/haiku-2026-11-15/openai-fallback/meter.jsonl` | 993 | `ac2400aa5624ee851b11e5ac7501d926c9a23adf6e71f48f4d1dce263633003b` |
| `benchmark/cost/results/haiku-2026-11-15/openai-fallback/oa_fallback.json` | 1,424 | `3e3a5f4594b33f6fc440efee6a8493099d2bfdbcfefbbae00fccbb600c55d449` |
