package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupName is the group name used in this package
const GroupName = "operator.config.grove.k8s.io"

var (
	// SchemeGroupVersion is group version used to register these objects
	SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: runtime.APIVersionInternal}
	// SchemeBuilder is used to add go types to the GroupVersionKind scheme
	SchemeBuilder      = runtime.NewSchemeBuilder(addKnownTypes)
	localSchemeBuilder = &SchemeBuilder
	// AddToScheme is a reference to the Scheme Builder's AddToScheme function.
	AddToScheme = localSchemeBuilder.AddToScheme
)

func init() {
	localSchemeBuilder.Register(addDefaultingFuncs, addKnownTypes)
}

func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&OperatorConfiguration{},
	)
	return nil
}

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}
