package main

import (
	"flag"
	"os"

	configv1alpha1 "github.com/NVIDIA/grove/operator/api/config/v1alpha1"
	groveopts "github.com/NVIDIA/grove/operator/cmd/opts"
	grovelogger "github.com/NVIDIA/grove/operator/internal/logger"
	grovemgr "github.com/NVIDIA/grove/operator/internal/manager"
	groveversion "github.com/NVIDIA/grove/operator/internal/version"

	ctrl "sigs.k8s.io/controller-runtime"
)

var (
	logger = ctrl.Log.WithName("grove")
)

func main() {
	ctx := ctrl.SetupSignalHandler()
	ctrl.SetLogger(grovelogger.MustNewLogger(false, configv1alpha1.InfoLevel, configv1alpha1.LogFormatJSON))

	cliFlagSet := flag.CommandLine
	groveversion.AddFlags(cliFlagSet)
	cliOpts := groveopts.NewCLIOptions(cliFlagSet)

	// parse and print command line flags
	flag.Parse()
	groveversion.PrintVersionAndExitIfRequested()

	logger.Info("Starting grove operator", "version", groveversion.Get())
	printFlags()

	operatorCfg, err := initializeOperatorConfig(cliOpts)
	if err != nil {
		logger.Error(err, "failed to initialize operator configuration")
		os.Exit(1)
	}

	mgr, err := grovemgr.CreateAndInitializeManager(operatorCfg)
	if err != nil {
		logger.Error(err, "failed to create grove controller manager")
		os.Exit(1)
	}
	logger.Info("Starting manager")
	if err = mgr.Start(ctx); err != nil {
		logger.Error(err, "Error running manager")
		os.Exit(1)
	}
}

func initializeOperatorConfig(cliOpts *groveopts.CLIOptions) (*configv1alpha1.OperatorConfiguration, error) {
	// complete and validate operator configuration
	if err := cliOpts.Complete(); err != nil {
		return nil, err
	}
	if err := cliOpts.Validate(); err != nil {
		return nil, err
	}
	return cliOpts.Config, nil
}

func printFlags() {
	var flagKVs []interface{}
	flag.VisitAll(func(f *flag.Flag) {
		flagKVs = append(flagKVs, f.Name, f.Value.String())
	})
	logger.Info("Running with flags", flagKVs...)
}
