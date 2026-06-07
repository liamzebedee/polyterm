manyterm
========

Terminal workspaces for multitasking devs.

- Store your terminal layout in a text file, that you can commit into git and open the same on your other devices
- Re-open the same Claude code instances for different projects
- Work with your locally-developed software setups without long-lived Terminal instances

## Usage

```
$ mtm term.workspace
```

[Manyterm running a full-stack workspace](demo/screenshot.png)

`term.workspace`:

```sh
workspaces
	Frontend
		"web"	~/app/web	npm run dev
		"storybook"	~/app/web	npm run storybook
		"claude"	~/app/web	claude
	Backend
		"api"	~/app/api	cargo watch -x run
		"worker"	~/app/api	cargo run --bin worker
		"claude"	~/app/api	claude
	Infra
		"postgres"	~/app	docker compose up db
		"redis"	~/app	redis-server
		"claude"	~/app	claude
	Tools
		"logs"	~/app	tail -f log/dev.log
		"claude"	~/app	claude
```
