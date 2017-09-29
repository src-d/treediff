# Tree Difference

This is the ongoing research on the Abstract Syntax Tree difference algorithm.

Currently, the code is written in Python, but the final solution will be incorporated into [libuast](https://github.com/bblfsh/libuast).

Plan:

1. Run regular sequence diff
2. Improve the Myers diff with UAST nodes count trick
3. Filter the nodes which are involved in the changes by lines
4. Hash them with the robust subtree sampling hash 
5. Map them using Linear Assignment LP
6. Compose the edit script
7. Output JSON
8. Visualize JSON in html


