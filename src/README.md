### Export Environment
```
conda env export | grep -v "^prefix: " > src/environment_droplet.yml
```

### Create Environment
```
conda env create -f src/environment_droplet.yml
```