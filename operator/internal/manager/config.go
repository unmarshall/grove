package manager

import (
	"flag"
	gctrl "github.com/NVIDIA/grove/operator/internal/controller"
)

const (
	// CLI Flag names
	enableLeaderElectionFlagName = "enable-leader-election"

	// CLI flag default values
	defaultEnableLeaderElection = false
)

// WebhookServerConfig is the configuration for the Webhook HTTPS server.
type WebhookServerConfig struct {
	Address
	// TLSArtifactsMountPath is the path where the TLS artifacts are mounted for the HTTPS server.
	TLSArtifactsMountPath string
}

type ServerConfig struct {
	// Webhook is the configuration for the HTTPS webhook server.
	Webhook WebhookServerConfig
	// Metrics is the configuration for serving the metrics endpoint.
	Metrics *Address
}

// Address encapsulates the bind address and port.
type Address struct {
	BindAddress string
	Port        int
}

// LeaderElectionConfig defines the configuration for the leader election for the controller manager.
type LeaderElectionConfig struct {
	// Enabled specifies whether to enable leader election for controller manager.
	Enabled bool
	// ID is the name of the resource that leader election will use for holding the leader lock.
	ID string
}

// Config defines the configuration for the controller manager.
type Config struct {
	// Server is the configuration for the HTTP(s) endpoints exposed by the manager.
	Server *ServerConfig
	// LeaderElection is the configuration for the leader election.
	LeaderElection LeaderElectionConfig
	// Controllers is the configuration for the grove controllers.
	Controllers *gctrl.Config
}

func (cfg *Config) InitFromFlags(fs *flag.FlagSet) {
	cfg.initializeEmptyConfig()
	flag.BoolVar(&cfg.LeaderElection.Enabled, enableLeaderElectionFlagName, defaultEnableLeaderElection,
		"Enable leader election for grove controllers. Enabling this will ensure that there is only one active set of grove controllers reconciling events.")
}

func (cfg *Config) initializeEmptyConfig() {
	cfg.Server = &ServerConfig{}
	cfg.Server.Webhook = WebhookServerConfig{}
	cfg.Server.Metrics = &Address{}
	cfg.LeaderElection = LeaderElectionConfig{}
	cfg.Controllers = &gctrl.Config{}
}
